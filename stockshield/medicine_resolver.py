"""Global Medicine Retrieval & Resolution Engine (GM-RRE).

A search-time federated resolver for medicines and pharmaceutical products.
It does NOT crawl the entire web or claim that every web page is authoritative.
Instead it queries multiple authoritative medicine registries in parallel, merges
identity evidence, scores agreement, and optionally uses a web-search index as a
last-mile discovery layer.

Facility inventory/demand is deliberately not inferred from public web data.
"""
from __future__ import annotations
import html, json, os, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from . import live_sources

CACHE = Path(os.getenv("STOCKSHIELD_RESOLVER_CACHE", "live_cache/resolver"))
CACHE.mkdir(parents=True, exist_ok=True)
TTL = int(os.getenv("STOCKSHIELD_RESOLVER_TTL", "3600"))
UA = "StockShield-AI/4.0 (global medicine resolver)"
LOCK = Lock()

SOURCES = {
    "rxnorm": "https://rxnav.nlm.nih.gov/REST/approximateTerm.json",
    "rxnorm_drugs": "https://rxnav.nlm.nih.gov/REST/drugs.json",
    "fda_ndc": "https://api.fda.gov/drug/ndc.json",
    "fda_labels": "https://api.fda.gov/drug/label.json",
    "dailymed": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
    "pubchem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "duckduckgo": "https://html.duckduckgo.com/html/",
}


def _clean(q):
    q = html.unescape(str(q or "")).strip()
    return re.sub(r"\s+", " ", q)[:180]


def _key(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:100] or "query"


def _get(url, params=None, headers=None, timeout=4):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    h = {"User-Agent": UA, "Accept": "application/json, text/html;q=0.9"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace"), r.headers.get("Content-Type", "")


def _json(url, params=None, headers=None, timeout=4):
    raw, _ = _get(url, params, headers, timeout)
    return json.loads(raw)


def _cached(name, fn, force=False):
    p = CACHE / f"{name}.json"
    if not force and p.exists() and time.time() - p.stat().st_mtime < TTL:
        try: return {"ok": True, "cached": True, "data": json.loads(p.read_text())}
        except Exception: pass
    try:
        data = fn(); p.write_text(json.dumps(data, ensure_ascii=False)); return {"ok": True, "cached": False, "data": data}
    except Exception as e:
        return {"ok": False, "cached": False, "error": str(e), "data": None}


def rxnorm(q):
    return _cached("rx_" + _key(q), lambda: _json(SOURCES["rxnorm"], {"term": q, "maxEntries": 25}))


def rxnorm_drugs(q):
    return _cached("rxd_" + _key(q), lambda: _json(SOURCES["rxnorm_drugs"], {"name": q}))


def fda_ndc(q):
    def call():
        s = f'openfda.generic_name:"{q.upper()}" OR openfda.brand_name:"{q.upper()}" OR generic_name:"{q.upper()}" OR brand_name:"{q.upper()}"'
        return _json(SOURCES["fda_ndc"], {"search": s, "limit": 100})
    return _cached("ndc_" + _key(q), call)


def fda_labels(q):
    def call():
        return _json(SOURCES["fda_labels"], {"search": f'openfda.generic_name:"{q.upper()}" OR openfda.brand_name:"{q.upper()}"', "limit": 20})
    return _cached("lbl_" + _key(q), call)


def dailymed(q):
    # DailyMed's v2 service is search-oriented and complements FDA/NLM identity.
    return _cached("dm_" + _key(q), lambda: _json(SOURCES["dailymed"], {"drug_name": q, "page": 1, "page_size": 50}))


def pubchem(q):
    path = urllib.parse.quote(q, safe="")
    return _cached("pc_" + _key(q), lambda: _json(f"{SOURCES['pubchem']}/{path}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"))


class _LinkParser:
    def __init__(self): self.links=[]
    def feed(self, s):
        from html.parser import HTMLParser
        class P(HTMLParser):
            def __init__(self, outer): super().__init__(); self.outer=outer; self.href=None; self.text=[]
            def handle_starttag(self, tag, attrs):
                if tag=='a': self.href=dict(attrs).get('href'); self.text=[]
            def handle_data(self, data):
                if self.href: self.text.append(data)
            def handle_endtag(self, tag):
                if tag=='a' and self.href:
                    t=re.sub(r'\s+',' ',''.join(self.text)).strip()
                    if t: self.outer.links.append((t,self.href))
                    self.href=None; self.text=[]
        P(self).feed(s)


def web_search(q):
    key=os.getenv("STOCKSHIELD_BRAVE_API_KEY")
    if key:
        def brave(): return _json(SOURCES["brave"], {"q": q, "count": 20, "search_lang":"en"}, {"X-Subscription-Token": key})
        got=_cached("brave_"+_key(q), brave)
        rows=[]
        for r in ((got.get("data") or {}).get("web",{}).get("results",[]) if got.get("ok") else []):
            rows.append({"title":r.get("title"),"url":r.get("url"),"description":r.get("description"),"source":"Brave Search"})
        return rows, got
    def ddg():
        raw,_=_get(SOURCES["duckduckgo"], {"q": q}, timeout=5)
        p=_LinkParser(); p.feed(raw)
        return [{"title":t,"url":u,"description":"Search-index discovery result","source":"DuckDuckGo"} for t,u in p.links[:20] if u.startswith("http")]
    got=_cached("web_"+_key(q), ddg)
    return (got.get("data") or []) if got.get("ok") else [], got


def _arr(data, *path):
    x=data
    for p in path:
        x=x.get(p,{}) if isinstance(x,dict) else {}
    return x if isinstance(x,list) else []


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _candidate(name, source, **kw):
    return {"name": str(name or "").strip(), "source": source, **kw}


def _parse(q, results):
    out=[]
    rx=results.get("rxnorm",{}).get("data") or {}
    for r in _arr(rx,"approximateGroup","candidate"):
        out.append(_candidate(r.get("name"),"NLM RxNorm",rxcui=r.get("rxcui"),score=float(r.get("score") or 0)/100 if str(r.get("score") or "").replace('.','',1).isdigit() else None))
    rd=results.get("rxnorm_drugs",{}).get("data") or {}
    for c in _arr(rd,"drugGroup","conceptGroup"):
        for r in c.get("conceptProperties",[]) or []: out.append(_candidate(r.get("name"),"NLM RxNorm",rxcui=r.get("rxcui")))
    ndc=results.get("fda_ndc",{}).get("data") or {}
    for r in ndc.get("results",[]) if isinstance(ndc,dict) else []:
        of=r.get("openfda") or {}; names=(of.get("generic_name") or [])+(of.get("brand_name") or [])
        nm=(r.get("generic_name") or r.get("brand_name") or (names[0] if names else ""))
        out.append(_candidate(nm,"FDA NDC",product_ndc=r.get("product_ndc"),brand_name=r.get("brand_name"),generic_name=r.get("generic_name"),dosage_form=r.get("dosage_form"),route=r.get("route"),manufacturer=r.get("labeler_name"),marketing_status=r.get("marketing_status"),rxcui=(of.get("rxcui") or [None])[0]))
    lbl=results.get("fda_labels",{}).get("data") or {}
    for r in lbl.get("results",[]) if isinstance(lbl,dict) else []:
        of=r.get("openfda") or {}; names=(of.get("generic_name") or [])+(of.get("brand_name") or [])
        if names: out.append(_candidate(names[0],"FDA Label",brand_name=(of.get("brand_name") or [None])[0],generic_name=(of.get("generic_name") or [None])[0],rxcui=(of.get("rxcui") or [None])[0],set_id=r.get("set_id")))
    dm=results.get("dailymed",{}).get("data") or {}
    for r in dm.get("data",[]) if isinstance(dm,dict) else []:
        out.append(_candidate(r.get("title") or r.get("drug_name"),"DailyMed",set_id=r.get("setid") or r.get("set_id"),published=r.get("published_date")))
    pc=results.get("pubchem",{}).get("data") or {}
    props=(pc.get("PropertyTable",{}).get("Properties",[]) if isinstance(pc,dict) else [])
    for r in props[:5]: out.append(_candidate(q,"PubChem",cid=r.get("CID"),molecular_formula=r.get("MolecularFormula"),molecular_weight=r.get("MolecularWeight"),iupac_name=r.get("IUPACName")))
    return out


def resolve(query, limit=20, force=False, web=True):
    q=_clean(query)
    if len(q)<2: raise ValueError("medicine search requires at least 2 characters")
    funcs={"rxnorm":lambda:rxnorm(q),"rxnorm_drugs":lambda:rxnorm_drugs(q),"fda_ndc":lambda:fda_ndc(q),"fda_labels":lambda:fda_labels(q),"dailymed":lambda:dailymed(q),"pubchem":lambda:pubchem(q)}
    results={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(fn):k for k,fn in funcs.items()}
        for f in as_completed(futs):
            k=futs[f]
            try: results[k]=f.result()
            except Exception as e: results[k]={"ok":False,"error":str(e)}
    candidates=_parse(q,results)
    nq=_norm(q)
    merged={}
    for c in candidates:
        name=c.get("name") or q; nk=_norm(name); key=nk or nq
        exact=(nk==nq or nq in nk or nk in nq)
        e=merged.setdefault(key,{"name":name,"sources":set(),"evidence":[],"match_score":0.0})
        e["sources"].add(c["source"]); e["evidence"].append({k:v for k,v in c.items() if k not in ("name","source")}); e["match_score"]=max(e["match_score"],1.0 if nk==nq else .85 if exact else .55)
    rows=[]
    for e in merged.values():
        src_bonus=min(.3,len(e["sources"])*.07); e["match_score"]=round(min(0.999,e["match_score"]+src_bonus),3); e["sources"]=sorted(e["sources"]); e["evidence_count"]=len(e["evidence"]); rows.append(e)
    rows.sort(key=lambda x:(-x["match_score"],-len(x["sources"]),x["name"]))
    web_rows=[]; web_meta=None
    if web and not rows:
        web_rows,web_meta=web_search(q)
    return {"query":q,"generated_at":datetime.now(timezone.utc).isoformat(),"engine":"GM-RRE v1","mode":"federated_authoritative_retrieval","exact_match":bool(rows and rows[0]["match_score"]>=.9),"results":rows[:max(1,min(int(limit),50))],"web_discovery":web_rows[:20],"source_status":{k:{"ok":bool(v.get("ok")),"cached":bool(v.get("cached")),"error":v.get("error")} for k,v in results.items()},"sources":SOURCES,"policy":"Public registries identify medicines/products; facility inventory, sales and stock remain authoritative local data."}
