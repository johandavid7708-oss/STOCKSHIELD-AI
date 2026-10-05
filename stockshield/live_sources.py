"""Live public-data intelligence connectors for StockShield AI.

These connectors deliberately separate public context signals from facility sales/inventory.
Public sources can inform supply/health context, but cannot substitute for authorized
pharmacy transaction data.
"""
from __future__ import annotations
import json, os, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path(os.getenv("STOCKSHIELD_LIVE_CACHE", "live_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TTL = int(os.getenv("STOCKSHIELD_LIVE_TTL", "900"))
NETWORK_BACKOFF_SECONDS = int(os.getenv("STOCKSHIELD_LIVE_BACKOFF", "30"))
NETWORK_BACKOFF_UNTIL = 0.0
NETWORK_LOCK = Lock()
UA = "StockShield-AI/3.0 (+https://localhost; public-data-intelligence)"

BENCHMARK_URL = "https://raw.githubusercontent.com/ismailsobhy2/Pharma-Sales-Time-Series-Analysis/master/sales_daily.xlsx.csv"
BENCHMARK_MAP = {"paracetamol": "N02BE", "acetaminophen": "N02BE", "cetirizine": "R06", "salbutamol": "R03", "albuterol": "R03", "analgesic": "N02BE", "respiratory": "R03", "antihistamine": "R06"}

SOURCES = {
    "fda_shortages": "https://api.fda.gov/drug/shortages.json",
    "fda_labels": "https://api.fda.gov/drug/label.json",
    "fda_recalls": "https://api.fda.gov/drug/enforcement.json",
    "rxnorm": "https://rxnav.nlm.nih.gov/REST/drugs.json",
    "cdc_positivity": "https://data.cdc.gov/resource/seuz-s2cv.json",
    "cdc_ed_respiratory": "https://data.cdc.gov/resource/vjzj-u7u8.json",
    "fda_ndc": "https://api.fda.gov/drug/ndc.json",
    "rxnorm_version": "https://rxnav.nlm.nih.gov/REST/version.json",
}


def _get(url, params=None, timeout=2):
    global NETWORK_BACKOFF_UNTIL
    # Serialize the first network attempt so a DNS outage does not fan out into
    # five simultaneous 2-second failures for every medicine request.
    with NETWORK_LOCK:
        if time.time() < NETWORK_BACKOFF_UNTIL:
            raise RuntimeError("live network temporarily unavailable; circuit breaker active")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            NETWORK_BACKOFF_UNTIL = time.time() + NETWORK_BACKOFF_SECONDS
            raise


def _cached(name, fn, force=False):
    p = CACHE_DIR / f"{name}.json"
    now = time.time()
    if not force and p.exists() and now - p.stat().st_mtime < TTL:
        try:
            cached_data = json.loads(p.read_text())
            if isinstance(cached_data, dict) and "_error" in cached_data:
                return {"ok": False, "cached": False, "fetched_at": cached_data.get("_at"), "error": cached_data.get("_error"), "data": None}
            return {"ok": True, "cached": True, "fetched_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(), "data": cached_data}
        except Exception:
            pass
    try:
        data = fn(); p.write_text(json.dumps(data, ensure_ascii=False));
        return {"ok": True, "cached": False, "fetched_at": datetime.now(timezone.utc).isoformat(), "data": data}
    except Exception as e:
        stale = None
        if p.exists():
            try: stale = json.loads(p.read_text())
            except Exception: pass
        # Negative-cache outages too, preventing repeated UI requests from hanging.
        if stale is None:
            try: p.write_text(json.dumps({"_error": str(e), "_at": datetime.now(timezone.utc).isoformat()}))
            except Exception: pass
        return {"ok": False, "cached": bool(stale), "fetched_at": datetime.now(timezone.utc).isoformat(), "error": str(e), "data": stale}


def _term(name):
    # Remove strength/form noise while retaining generic/brand search terms.
    return re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%)\b", "", name, flags=re.I).strip()


def fda_shortage(name, force=False):
    term = _term(name).upper()
    def call():
        q = f'generic_name:"{term}" OR proprietary_name:"{term}" OR openfda.generic_name:"{term}" OR openfda.brand_name:"{term}"'
        return _get(SOURCES["fda_shortages"], {"search": q, "limit": 20, "sort": "update_date:desc"})
    return _cached("shortage_" + re.sub(r"\W+", "_", term)[:70], call, force)


def fda_label(name, force=False):
    term = _term(name).upper()
    def call():
        return _get(SOURCES["fda_labels"], {"search": f'openfda.generic_name:"{term}"', "limit": 5})
    return _cached("label_" + re.sub(r"\W+", "_", term)[:70], call, force)


def fda_recalls(name, force=False):
    term = _term(name).upper()
    def call():
        return _get(SOURCES["fda_recalls"], {"search": f'product_description:"{term}"', "limit": 20, "sort": "report_date:desc"})
    return _cached("recall_" + re.sub(r"\W+", "_", term)[:70], call, force)


def rxnorm(name, force=False):
    term = _term(name)
    return _cached("rxnorm_" + re.sub(r"\W+", "_", term.lower())[:70], lambda: _get(SOURCES["rxnorm"], {"name": term}), force)


def cdc_positivity(force=False):
    def call():
        return _get(SOURCES["cdc_positivity"], {"$limit": 1000, "$order": "week_end DESC"})
    return _cached("cdc_positivity", call, force)


def cdc_respiratory(force=False):
    def call():
        return _get(SOURCES["cdc_ed_respiratory"], {"$limit": 1000, "$order": "date DESC"})
    return _cached("cdc_ed_respiratory", call, force)


def benchmark_trend(name, category, force=False):
    key = "benchmark_pharma_sales"
    def call():
        with urllib.request.urlopen(urllib.request.Request(BENCHMARK_URL, headers={"User-Agent": UA}), timeout=4) as r:
            return r.read().decode("utf-8")
    got = _cached(key, call, force)
    text = got.get("data") if isinstance(got.get("data"), str) else None
    code = next((v for k,v in BENCHMARK_MAP.items() if k in (name+" "+category).lower()), None)
    if not text or not code:
        return {"available": False, "code": code, "reason": "No mapped ATC category or benchmark unavailable", "source": BENCHMARK_URL, "fetch": got}
    import csv, io, statistics
    rows=[]
    try:
        for r in csv.DictReader(io.StringIO(text)):
            try: rows.append((datetime.strptime(r["datum"], "%m/%d/%Y").date(), float(r.get(code) or 0)))
            except Exception: pass
    except Exception as e:
        return {"available": False, "code": code, "reason": str(e), "source": BENCHMARK_URL}
    rows.sort(); vals=[v for _,v in rows]
    if len(vals)<60: return {"available": False, "code": code, "reason":"insufficient benchmark history", "source":BENCHMARK_URL}
    recent=statistics.mean(vals[-30:]); prior=statistics.mean(vals[-395:-30]) if len(vals)>395 else statistics.mean(vals[:-30]); growth=(recent/prior-1) if prior else 0
    return {"available": True, "code": code, "start": rows[0][0].isoformat(), "end": rows[-1][0].isoformat(), "records": len(rows), "recent_30d_mean": round(recent,3), "prior_mean": round(prior,3), "growth_pct": round(growth*100,2), "direction": "increasing" if growth>.08 else "decreasing" if growth<-.08 else "stable", "source": BENCHMARK_URL, "note":"Public pharmacy benchmark history; not current facility demand."}


def catalogue_lookup(term, limit=20, force=False):
    """Fetch current identity/SKU evidence for a medicine search from NLM RxNorm + FDA NDC.

    This is intentionally search-scoped rather than a 4,000-request startup crawl. The
    local catalogue remains the fast searchable index; live lookup enriches the exact
    medicine the user is interested in.
    """
    term = _term(term)
    safe = re.sub(r"\W+", "_", term.lower())[:70]
    rx = rxnorm(term, force)
    ndc = _cached("ndc_search_" + safe, lambda: _get(SOURCES["fda_ndc"], {"search": f'openfda.generic_name:"{term.upper()}"', "limit": min(int(limit), 100)}), force)
    concepts = []
    data = (rx.get("data") or {}) if isinstance(rx, dict) else {}
    groups = data.get("minConceptGroup", {}).get("minConcept", []) if isinstance(data, dict) else []
    for c in groups[:limit]:
        concepts.append({"name": c.get("name"), "rxcui": c.get("rxcui"), "source": "NLM RxNorm", "source_url": SOURCES["rxnorm"]})
    ndc_rows = _results(ndc)
    ndc_out = [{"product_ndc": r.get("product_ndc"), "generic_name": r.get("generic_name"), "brand_name": r.get("brand_name"), "dosage_form": r.get("dosage_form"), "active_ingredients": r.get("active_ingredients"), "marketing_status": r.get("marketing_status"), "source": "FDA NDC", "source_url": SOURCES["fda_ndc"]} for r in ndc_rows[:limit]]
    return {"query": term, "generated_at": datetime.now(timezone.utc).isoformat(), "rxnorm": concepts, "fda_ndc": ndc_out, "sources": {"rxnorm": SOURCES["rxnorm"], "fda_ndc": SOURCES["fda_ndc"]}}


def _results(x):
    if not x: return []
    data = x.get("data")
    return data.get("results", []) if isinstance(data, dict) else []


def _trend(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 4: return {"direction": "unknown", "recent": None, "baseline": None, "ratio": None}
    recent = sum(vals[:4]) / 4
    baseline = sum(vals[4:16]) / max(len(vals[4:16]), 1) if len(vals) > 4 else recent
    ratio = recent / baseline if baseline else 1.0
    direction = "increasing" if ratio >= 1.08 else "decreasing" if ratio <= .92 else "stable"
    return {"direction": direction, "recent": recent, "baseline": baseline, "ratio": ratio}


def build_live_intelligence(medicine, force=False):
    name, category = medicine["name"], (medicine.get("category") or "")
    jobs = {
        "shortage": lambda: fda_shortage(name, force),
        "label": lambda: fda_label(name, force),
        "recalls": lambda: fda_recalls(name, force),
        "rx": lambda: rxnorm(name, force),
        "pos": lambda: cdc_positivity(force),
        "benchmark": lambda: benchmark_trend(name, category, force),
    }
    out = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    shortage, label, recalls, rx, pos, benchmark = (out[k] for k in ("shortage", "label", "recalls", "rx", "pos", "benchmark"))

    term = _term(name).upper()
    short_rows = _results(shortage)
    recall_rows = _results(recalls)
    shortage_hits = [{"status": r.get("availability") or r.get("status") or "reported", "availability": r.get("availability"), "shortage_status": r.get("status"), "reason": r.get("shortage_reason") or r.get("change_reason"), "update_date": r.get("update_date") or r.get("change_date"), "generic_name": r.get("openfda", {}).get("generic_name", []) or r.get("generic_name", []), "company": r.get("company_name"), "presentation": r.get("presentation")} for r in short_rows]
    active_shortages = [r for r in shortage_hits if str(r.get("shortage_status") or "").lower() in ("current", "active", "ongoing") or str(r.get("availability") or "").lower() not in ("available", "") ]
    supply_pressure = min(1.0, len(active_shortages) / 3.0)
    recall_signal = min(1.0, len(recall_rows) / 3.0)

    positivity = {}
    for r in _results(pos):
        pathogen = r.get("pathogen")
        if pathogen:
            try: positivity.setdefault(pathogen, []).append(float(r.get("percent_test_positivity") or 0))
            except (TypeError, ValueError): pass
    respiratory = {k: _trend(v) for k, v in positivity.items()}
    relevant = any(x in (category + " " + name).lower() for x in ("respir", "influenza", "flu", "antiviral", "salbutamol", "oseltamivir"))
    respiratory_pressure = max([max(0.0, min(.25, (v.get("ratio") or 1)-1)) for v in respiratory.values()] or [0.0]) if relevant else 0.0
    allr = (shortage, label, recalls, rx, pos)
    freshness = [x.get("fetched_at") for x in allr if x]
    live_ok = any(x.get("ok") for x in allr)
    return {"medicine_id": medicine["id"], "medicine": name, "generated_at": datetime.now(timezone.utc).isoformat(), "live": live_ok, "data_mode": "LIVE_PUBLIC_CONTEXT" if live_ok else "CACHED_OR_UNAVAILABLE",
        "supply": {"shortage_matches": len(active_shortages), "matched_records": len(shortage_hits), "pressure": round(supply_pressure, 3), "records": shortage_hits[:5], "source": SOURCES["fda_shortages"], "status": "signal detected" if shortage_hits else "no matching public shortage record returned"},
        "safety": {"recall_matches": len(recall_rows), "pressure": round(recall_signal, 3), "source": SOURCES["fda_recalls"]},
        "respiratory_context": {"relevant": relevant, "pressure": round(respiratory_pressure, 3), "pathogens": respiratory, "source": SOURCES["cdc_positivity"]},
        "historical_benchmark": benchmark,
        "identity": {"rxnorm_available": bool(_results(rx)), "label_matches": len(_results(label)), "source": SOURCES["rxnorm"]},
        "source_meta": {"fda_shortages_last_updated": (shortage.get("data") or {}).get("meta", {}).get("last_updated") if isinstance(shortage.get("data"), dict) else None, "fda_labels_last_updated": (label.get("data") or {}).get("meta", {}).get("last_updated") if isinstance(label.get("data"), dict) else None, "fda_recalls_last_updated": (recalls.get("data") or {}).get("meta", {}).get("last_updated") if isinstance(recalls.get("data"), dict) else None},
        "provenance": {"sources": SOURCES, "freshness": freshness, "note": "Public signals are contextual evidence. They do not represent facility inventory or patient-level demand."}}


def _probe(name, url, params=None, force=False):
    return _cached("probe_" + name, lambda: _get(url, params or {"limit": 1}), force)


def system_snapshot(force=False):
    jobs = {
        "fda_shortages": lambda: _probe("fda_shortages", SOURCES["fda_shortages"], {"limit": 1}, force),
        "fda_recalls": lambda: _probe("fda_recalls", SOURCES["fda_recalls"], {"limit": 1, "sort": "report_date:desc"}, force),
        "fda_ndc": lambda: _probe("fda_ndc", SOURCES["fda_ndc"], {"limit": 1}, force),
        "rxnorm": lambda: _probe("rxnorm", SOURCES["rxnorm"], {"name": "amoxicillin"}, force),
        "rxnorm_version": lambda: _probe("rxnorm_version", SOURCES["rxnorm_version"], {}, force),
        "cdc_positivity": cdc_positivity,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": {k: {"ok": v.get("ok"), "cached": v.get("cached"), "fetched_at": v.get("fetched_at"), "error": v.get("error"), "source_url": SOURCES.get(k)} for k,v in results.items()}, "source_urls": SOURCES, "refresh_policy": {"live_ttl_seconds": TTL, "note": "Force refresh bypasses the local TTL. Cached data is explicitly labeled and never presented as fresh."}}
