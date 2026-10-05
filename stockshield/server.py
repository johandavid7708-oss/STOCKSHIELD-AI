"""Dependency-free JSON API + static dashboard (stdlib http.server) over the StockShield core."""
import hashlib, json, logging, os, re, sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np
import pandas as pd
from . import core, intel, live_sources, live_model, medicine_resolver

DB = os.getenv("STOCKSHIELD_DB", "stockshield.db"); WEB = os.path.join(os.path.dirname(__file__), "web"); log = logging.getLogger("stockshield")


class HttpError(Exception):
    def __init__(self, code, msg): self.code, self.msg = code, msg


def q(sql, args=()):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    try: return [dict(r) for r in con.execute(sql, args)]
    finally: con.close()


def ev(key):
    r = q("SELECT value FROM evaluation WHERE key=?", (key,)); return json.loads(r[0]["value"]) if r else None


def med(s):
    if not re.fullmatch(r"[0-9]{1,9}", s): raise HttpError(400, "medicine id must be an integer")
    m = q("SELECT * FROM medicine WHERE id=?", (int(s),))
    if not m: raise HttpError(404, "medicine not found")
    return m[0]


def overview():
    fc = {}
    for r in q("SELECT * FROM forecast ORDER BY medicine_id,forecast_date"): fc.setdefault(r["medicine_id"], []).append(r["predicted_demand"])
    risk = {r["medicine_id"]: r for r in q("SELECT * FROM risk_prediction")}
    stock = {r["medicine_id"]: r["quantity_available"] for r in q("SELECT medicine_id,quantity_available FROM inventory WHERE (medicine_id,date) IN (SELECT medicine_id,MAX(date) FROM inventory GROUP BY medicine_id)")}
    out = []
    for m in q("SELECT * FROM medicine"):
        r, f = risk.get(m["id"], {}), fc.get(m["id"], [0] * 14)
        base = {**m, "stock": stock.get(m["id"]), "forecast_7d": sum(f[:7]), "forecast_14d": sum(f), **{k: r.get(k) for k in ("risk_probability", "risk_level", "days_remaining", "expected_stockout_date")}}
        try:
            live = live_sources.build_live_intelligence(m, force=False)
            adj, overlay = live_model.apply_forecast([{"predicted_demand": x} for x in f], live, m.get("category", ""), m.get("name", ""))
            af = [x["predicted_demand"] for x in adj]
            daily = sum(af[:7]) / 7 if af else 0
            p = live_model.apply_risk(float(base.get("risk_probability") or 0), overlay)
            # Recompute shortage timing against the live-context demand path.
            days, day = core.shortage(float(base["stock"] or 0), af)
            base.update({"forecast_7d": sum(af[:7]), "forecast_14d": sum(af), "risk_probability": p, "risk_level": core.level(p),
                         "days_remaining": days, "expected_stockout_date": None if day is None else (pd.Timestamp.today().normalize() + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                         "live_context": overlay, "live_data_mode": live.get("data_mode")})
        except Exception:
            base["live_data_mode"] = "UNAVAILABLE"
        out.append(base)
    return out


def frame(mid):
    con = sqlite3.connect(DB)
    try: return core.process(*[pd.read_sql(f"SELECT * FROM {t} WHERE {k}=?", con, params=(mid,)) for t, k in (("medicine", "id"), ("sales", "medicine_id"), ("inventory", "medicine_id"))])
    finally: con.close()


def history(m, days=90):
    df = frame(m["id"]); df["ma7"] = df.demand_adj.rolling(7).mean(); df = df.tail(days)
    return {"medicine_id": m["id"], "days": [dict(date=r.date.strftime("%Y-%m-%d"), sold=r.quantity_sold, demand_adj=r.demand_adj, censored=bool(r.censored), inventory=r.quantity_available, ma7=None if pd.isna(r.ma7) else r.ma7) for r in df.itertuples()],
            "stockouts": q("SELECT * FROM stockout_event WHERE medicine_id=? ORDER BY start_date DESC", (m["id"],))}


def forecast(m):
    p = profile(m); rows = q("SELECT * FROM forecast WHERE medicine_id=? ORDER BY forecast_date", (m["id"],)); base = float(frame(m["id"]).demand_adj.tail(7).mean()); f = p["fc"];
    daily = []
    for i, r in enumerate(rows):
        daily.append({**r, **f[i]})
    return {"trend": p["trend"], "comparison": p["comparison"], "extended_30d": {"total": sum(x["predicted_demand"] for x in f) / 14 * 30, "label": "Extended planning estimate: the live-context-adjusted 14-day forecast rate extrapolated to 30 days. No 30-day model has been validated."}, "medicine_id": m["id"], "model_version": core.MODEL_VERSION, "daily": daily, "forecast_7d": sum(x["predicted_demand"] for x in f[:7]), "forecast_14d": sum(x["predicted_demand"] for x in f), "baseline_daily": base, "baseline_7d": base * 7, "baseline_14d": base * 14, "live_context": p["live_context"],
            "uncertainty": "Approximate validation-residual uncertainty band from the stored temporal hold-out, scaled by the bounded public-signal overlay; not a formal confidence interval."}


def risk(m):
    p = profile(m); r = p["risk"]
    return {**r, "drivers": r["explanation"].split(" | "), "lead_time_days": m["lead_time_days"], "safety_stock": m["safety_stock"], "alerts": q("SELECT type,severity,message FROM alert WHERE medicine_id=?", (m["id"],)), "live_context": p["live_context"], "base_risk_probability": p["base_risk_probability"]}




def live_intelligence(m):
    live = live_sources.build_live_intelligence(m, force=False)
    live["model_overlay"] = live_model.build_overlay(live, m.get("category", ""), m.get("name", ""))
    return live


def live_refresh(m=None):
    if m is not None:
        return live_sources.build_live_intelligence(m, force=True)
    return live_sources.system_snapshot(force=True)


def num(qs, k, lo, hi, default):
    try: v = float(qs.get(k, default))
    except ValueError: raise HttpError(400, f"{k} must be a number")
    if not lo <= v <= hi: raise HttpError(400, f"{k} must be between {lo} and {hi}")
    return v


REORDER_NOTE = "Recommended reorder quantity is based on projected demand, lead time and safety-stock requirements. It is a deterministic decision-support estimate, not guaranteed optimal procurement."
_cache = {}


def cached(key, fn):
    try: stamp = os.stat(DB).st_mtime_ns
    except OSError: stamp = 0
    k = (DB, stamp, key)
    if k not in _cache: _cache[k] = fn()
    return _cache[k]


def _cmp(fe, mid):
    out = []
    for e in fe:
        b = e["by_medicine"][str(mid)]; mo, ba = b["rf_mae"], b["moving_avg_mae"]
        out.append({"horizon": e["horizon"], "model_mae": mo, "baseline_mae": ba, "abs_improvement": ba - mo, "pct_improvement": 100 * (ba - mo) / ba if ba > 0 else None, "n_validation_rows": b["n"]})
    return out


def profile(m):
    def build():
        df = frame(m["id"]); base_ov = next(o for o in overview() if o["id"] == m["id"]); rows = q("SELECT forecast_date,predicted_demand,lower_bound,upper_bound,model_version FROM forecast WHERE medicine_id=? ORDER BY forecast_date", (m["id"],))
        base_fc = [dict(r) for r in rows]; live = live_sources.build_live_intelligence(m, force=False); fc, overlay = live_model.apply_forecast(base_fc, live, m.get("category", ""), m.get("name", ""))
        tr = intel.trend(df.demand_adj); ce = intel.censored(df); f7 = sum(x["predicted_demand"] for x in fc[:7]); lead = m["lead_time_days"]; sig = float((ev("forecast") or [{"sigma": 0}])[0]["sigma"])
        base_risk_row = q("SELECT * FROM risk_prediction WHERE medicine_id=?", (m["id"],))[0]; base_prob = float(base_risk_row["risk_probability"]); live_base_prob = core.shortfall_prob(base_ov["stock"], f7 / 7, lead, m["safety_stock"], sig); prob = live_model.apply_risk(live_base_prob, overlay); days, shortage_day = core.shortage(base_ov["stock"], [x["predicted_demand"] for x in fc])
        risk_row = {**base_risk_row, "risk_probability": prob, "risk_level": core.level(prob), "days_remaining": days, "expected_stockout_date": None if shortage_day is None else (pd.Timestamp.today().normalize() + pd.Timedelta(days=shortage_day)).strftime("%Y-%m-%d"), "explanation": base_risk_row["explanation"] + (f" | Public shortage/recall context adds a bounded risk adjustment of {overlay['risk_probability_bump']:.0%}." if overlay["risk_probability_bump"] > 0 else " | No bounded public shortage/recall risk adjustment applied.")}
        ro = intel.reorder(base_ov["stock"], m["safety_stock"], f7 / 7, lead, f7, tr["growth_pct"] + overlay["demand_shift_pct"]); pj = intel.projection(base_ov["stock"], m["safety_stock"], [x["predicted_demand"] for x in fc], lead)
        an = intel.anomalies(df, tr, ce)
        if overlay["risk_probability_bump"] > 0:
            an["items"].append({"level": "WATCH", "type": "LIVE_SUPPLY_SIGNAL", "message": f"Public supply/safety signals increased operational risk by {overlay['risk_probability_bump']:.0%}.", "why": [f"FDA shortage matches: {overlay['components']['shortage_matches']}", f"Public recall matches: {overlay['components']['recall_matches']}"]}); an["status"] = "WATCH" if an["status"] == "NORMAL" else an["status"]
        health = intel.health(prob, days, lead, tr, f7, sig, ce["censored_days_90d"] / 90)
        return dict(trend=tr, censored=ce, projection={**pj, "dates": [r["forecast_date"] for r in rows]}, reorder=ro, anomalies=an, health=health, ov={**base_ov, "risk_probability": prob, "risk_level": core.level(prob), "days_remaining": days, "expected_stockout_date": risk_row["expected_stockout_date"], "forecast_7d": f7, "forecast_14d": sum(x["predicted_demand"] for x in fc), "live_context": overlay}, fc=fc, comparison=_cmp(ev("forecast"), m["id"]), risk=risk_row, live_context=overlay, base_risk_probability=live_base_prob, live=live)
    return cached(("profile", m["id"]), build)

def _scen(m, fc, sig, stock, lead, qty, mult, safety, live_bump=0.0):
    f = [v["predicted_demand"] * mult if isinstance(v, dict) else v * mult for v in fc]; days, k = core.shortage(stock, f); f7 = sum(f[:7]); p = live_model.apply_risk(core.shortfall_prob(stock, f7 / 7, lead, safety, sig), {"risk_probability_bump": live_bump}); s, traj = stock, []
    for d, v in enumerate(f, 1):
        s -= v; s += qty if d == lead else 0; traj.append(round(s, 1))
    ro = intel.reorder(stock, safety, f7 / 7, lead, f7)
    return {"stock": stock, "lead_time_days": lead, "reorder_qty": qty, "demand_multiplier": mult, "safety_stock": safety, "coverage_days": days, "shortage_day": k, "shortage_before_replenishment": days < lead,
            "risk_probability": p, "risk_level": core.level(p), "projected_inventory": traj, "stockout_after_reorder": min(traj) < 0, "reorder_required": ro["reorder_required"], "reorder_point": ro["reorder_point"], "recommended_reorder": ro["recommended_quantity"]}


def simulate(m, qs):
    scenario=(qs.get("scenario") or "custom").lower().replace("-","_").replace(" ","_")
    p=profile(m); current_stock=p["ov"]["stock"]
    defaults={"stock":current_stock,"lead":m["lead_time_days"],"reorder_qty":0,"demand_mult":1,"safety":m["safety_stock"]}
    presets={
        "demand_surge":{"demand_mult":1.5}, "surge":{"demand_mult":1.5},
        "demand_drop":{"demand_mult":0.6}, "drop":{"demand_mult":0.6},
        "supplier_delay":{"lead":m["lead_time_days"]+5}, "delay":{"lead":m["lead_time_days"]+5},
        "seasonal_spike":{"demand_mult":1.3}, "seasonal":{"demand_mult":1.3},
        "stockout_recovery":{"stock":0,"reorder_qty":0}, "recovery":{"stock":0,"reorder_qty":0}
    }
    for k,v in presets.get(scenario,{}).items(): defaults[k]=v
    # Explicit query values override a preset.
    stock=num(qs,"stock",0,1e6,defaults["stock"]); lead=int(num(qs,"lead",1,60,defaults["lead"])); qty=num(qs,"reorder_qty",0,1e6,defaults["reorder_qty"])
    mult=num(qs,"demand_mult",.1,5,defaults["demand_mult"]); safety=num(qs,"safety",0,1e6,defaults["safety"])
    sig=ev("forecast")[0]["sigma"]
    sc=_scen(m,p["fc"],sig,stock,lead,qty,mult,safety,p["live_context"]["risk_probability_bump"]); base=_scen(m,p["fc"],sig,p["ov"]["stock"],m["lead_time_days"],0,1,m["safety_stock"],p["live_context"]["risk_probability_bump"])
    sc["scenario_name"]=scenario; sc["reasoning"]=f"Scenario changes demand by {mult:.0%}, supplier lead time to {lead} days, stock to {stock:.0f}, and safety stock to {safety:.0f}. The same forecast/risk/inventory equations are then recomputed."
    sc["recommended_action"]=f"Reorder {sc['recommended_reorder']} units now." if sc["reorder_required"] else "Monitor inventory; no immediate reorder is required."
    return {**sc,"scenario":sc,"current":base,"delta":{"risk_probability":sc["risk_probability"]-base["risk_probability"],"coverage_days":sc["coverage_days"]-base["coverage_days"],"recommended_reorder":sc["recommended_reorder"]-base["recommended_reorder"]}}


def catalogue(search=None, limit=100, offset=0, live=False):
    limit=max(1,min(int(limit),500)); offset=max(0,int(offset)); args=[]; where=""
    if search:
        where="WHERE name LIKE ? OR category LIKE ? OR manufacturer LIKE ?"; qv=f"%{search}%"; args=[qv,qv,qv]
    total=q(f"SELECT COUNT(*) n FROM medicine_catalogue {where}", args)[0]["n"]
    rows=q(f"SELECT * FROM medicine_catalogue {where} ORDER BY name, catalogue_id LIMIT ? OFFSET ?", args+[limit,offset])
    out={"total":total,"limit":limit,"offset":offset,"records":rows,"source":"local_real_medicine_reference","live_enrichment":"FDA NDC + NLM RxNorm available through live lookup for selected medicines"}
    if search and str(live).lower() in ("1","true","yes","on"):
        out["live_lookup"]=live_sources.catalogue_lookup(search, min(limit,20), force=False)
    return out

def medicine_search(query, limit=20, web=True, force=False):
    qv = str(query or "").strip()
    if len(qv) < 2:
        raise HttpError(400, "medicine search requires at least 2 characters")
    # Local catalogue is the fast first layer; federated live retrieval resolves
    # medicines that are not locally stored and attaches provenance/evidence.
    local = catalogue(qv, min(int(limit), 20), 0, False)
    live = medicine_resolver.resolve(qv, limit=limit, force=force, web=str(web).lower() not in ("0","false","no","off"))
    # Search-time learning cache: verified public registry results become local reference
    # records, so the catalogue grows organically without confusing reference records
    # with facility inventory.
    try:
        con = sqlite3.connect(DB)
        rows=[]
        for r in (live.get("results") or [])[:50]:
            name=str(r.get("name") or qv).strip()
            if not name: continue
            src="; ".join(r.get("sources") or [])
            cid="live_"+hashlib.sha256((src+"|"+name).encode()).hexdigest()[:24]
            evidence=r.get("evidence") or [{}]
            first=evidence[0] if evidence else {}
            rows.append((cid,name,"Live registry",first.get("dosage_form") or "",first.get("strength") or "",first.get("manufacturer") or "", "", "live_resolved", src))
        if rows:
            con.executemany("INSERT OR IGNORE INTO medicine_catalogue VALUES(?,?,?,?,?,?,?,?,?)", rows); con.commit()
        con.close()
    except Exception as e:
        log.warning("resolver cache write skipped: %s", e)
    # If live registries are unavailable, preserve useful local identity evidence instead of
    # collapsing the search into an empty state. This is explicitly marked LOCAL_REFERENCE.
    if not (live.get("results") or []) and local.get("records"):
        fallback=[]
        for rec in local.get("records", [])[:20]:
            fallback.append({
                "name": rec.get("name") or qv,
                "sources": ["StockShield Local Reference"],
                "evidence_count": 1,
                "match_score": 0.82 if str(rec.get("name") or "").lower()==qv.lower() else 0.62,
                "evidence": [{k: rec.get(k) for k in ("category","strength","dosage_form","manufacturer","classification") if rec.get(k) not in (None, "")}],
                "resolution_mode": "LOCAL_REFERENCE_FALLBACK"
            })
        live["results"] = fallback
        live["exact_match"] = bool(fallback and fallback[0]["match_score"] >= .8)
        live["mode"] = "local_reference_fallback"
        live["resolution_note"] = "Authoritative live registries were unavailable or returned no records; local reference evidence is shown explicitly and must not be interpreted as regulatory confirmation."

    # Build a second-layer intelligence packet from the highest-confidence identity.
    # This is intentionally public-context intelligence; it never fabricates facility stock.
    top = (live.get("results") or [None])[0]
    if top:
        evidence = top.get("evidence") or []
        category = ""
        for evd in evidence:
            if isinstance(evd, dict) and evd.get("dosage_form"):
                category = str(evd.get("dosage_form"))
                break
        try:
            public_intel = live_sources.build_live_intelligence({"id": "search", "name": top.get("name") or qv, "category": category}, force=force)
        except Exception as e:
            public_intel = {"live": False, "data_mode": "UNAVAILABLE", "error": str(e), "medicine": top.get("name") or qv}
        # A transparent, bounded evidence score. This is an evidence-quality score,
        # not a clinical safety score and not a probability of approval.
        source_count = len(top.get("sources") or [])
        evidence_count = int(top.get("evidence_count") or 0)
        identity_confidence = round(min(0.999, 0.55 + 0.07 * source_count + 0.025 * min(evidence_count, 10) + 0.20 * float(top.get("match_score") or 0)), 3)
        public_intel["identity_confidence"] = identity_confidence
        public_intel["source_agreement"] = source_count
        public_intel["evidence_records"] = evidence_count
        public_intel["interpretation"] = "Evidence-weighted public context; not a clinical recommendation and not facility inventory."
    else:
        public_intel = {"live": False, "data_mode": "NO_AUTHORITATIVE_MATCH", "identity_confidence": 0.0, "source_agreement": 0, "evidence_records": 0}
    return {"query": qv, "local_matches": local["records"], "local_total_matches": local["total"], "intelligence": public_intel, **live}

def monitoring_universe(search=None, limit=500, offset=0):
    """Return the complete medicine monitoring universe without fabricating facility telemetry.

    A medicine is analytics-ready only when it exists in the facility ``medicine`` table.
    Every catalogue identity is still monitored at the identity/reference layer.
    """
    limit=max(1,min(int(limit),1000)); offset=max(0,int(offset)); args=[]; where=""
    if search:
        where="WHERE c.name LIKE ? OR c.category LIKE ? OR c.manufacturer LIKE ? OR c.strength LIKE ?"
        qv=f"%{search}%"; args=[qv,qv,qv,qv]
    sql=f"""SELECT c.*, m.id AS facility_medicine_id,
              CASE WHEN m.id IS NULL THEN 'REFERENCE_ONLY' ELSE 'ANALYTICS_READY' END AS monitoring_status
              FROM medicine_catalogue c
              LEFT JOIN medicine_catalogue mc ON mc.catalogue_id=c.catalogue_id
              LEFT JOIN medicine m ON lower(trim(m.name))=lower(trim(c.name))
              {where}
              GROUP BY c.catalogue_id
              ORDER BY c.name, c.catalogue_id LIMIT ? OFFSET ?"""
    rows=q(sql,args+[limit,offset])
    total=q(f"SELECT COUNT(*) n FROM medicine_catalogue c {where}",args)[0]["n"]
    ready=q("SELECT COUNT(*) n FROM medicine")[0]["n"]
    return {"total":total,"analytics_ready":ready,"limit":limit,"offset":offset,"records":rows,"definition":"Monitored means present in the local medicine identity/reference universe. Analytics-ready means facility inventory/sales history exists; no stock, demand or forecast is fabricated for reference-only medicines."}


def medicine_intelligence(search=None, limit=50, offset=0, sort="priority"):
    """Portfolio-scale intelligence for every catalogue identity.

    Reference-only records receive identity/data-readiness/public-context analysis.
    Facility demand, stock, forecast, risk and reorder metrics are only attached when
    an authorized facility medicine identity exists. This endpoint never fabricates
    facility telemetry for catalogue-only medicines.
    """
    limit=max(1,min(int(limit),100)); offset=max(0,int(offset)); args=[]; where=""
    if search:
        where="WHERE c.name LIKE ? OR c.category LIKE ? OR c.manufacturer LIKE ? OR c.strength LIKE ? OR c.dosage_form LIKE ?"
        qv=f"%{search}%"; args=[qv,qv,qv,qv,qv]
    total=q(f"SELECT COUNT(*) n FROM medicine_catalogue c {where}",args)[0]["n"]
    # Portfolio summary is computed from catalogue facts, not invented demand/risk.
    counts=q(f"""SELECT
      COUNT(*) total,
      SUM(CASE WHEN COALESCE(name,'')<>'' THEN 1 ELSE 0 END) named,
      SUM(CASE WHEN COALESCE(strength,'')<>'' THEN 1 ELSE 0 END) with_strength,
      SUM(CASE WHEN COALESCE(dosage_form,'')<>'' THEN 1 ELSE 0 END) with_form,
      SUM(CASE WHEN COALESCE(manufacturer,'')<>'' THEN 1 ELSE 0 END) with_manufacturer,
      SUM(CASE WHEN COALESCE(category,'')<>'' THEN 1 ELSE 0 END) with_category,
      SUM(CASE WHEN COALESCE(indication,'')<>'' THEN 1 ELSE 0 END) with_indication
      FROM medicine_catalogue c {where}""",args)[0]
    facility=q("SELECT id,name,category FROM medicine")
    facility_by_name={str(x['name']).strip().lower():x for x in facility}
    rows=q(f"SELECT c.* FROM medicine_catalogue c {where} ORDER BY c.name,c.catalogue_id LIMIT ? OFFSET ?",args+[limit,offset])
    records=[]
    for c in rows:
        name=str(c.get('name') or '').strip(); key=name.lower()
        completeness=sum(bool(c.get(k)) for k in ('name','category','strength','dosage_form','manufacturer','indication'))/6
        fac=facility_by_name.get(key)
        if fac:
            # Attach genuine facility analytics only; do not calculate a fake value here.
            facility_state={'status':'ANALYTICS_READY','facility_medicine_id':fac['id'],
                             'detail_endpoint':f"/api/medicines/{fac['id']}",
                             'analytics_note':'Facility telemetry exists; open the medicine profile to calculate live forecast/risk/reorder metrics.'}
        else:
            facility_state={'status':'REFERENCE_ONLY','facility_medicine_id':None,'data_boundary':'Facility inventory, sales, supplier lead time and patient demand are not available for this identity.'}
        # Evidence quality, not medical/clinical risk.
        evidence_score=round(completeness*100)
        records.append({
            'catalogue_id':c['catalogue_id'],'name':name,'category':c.get('category'),'dosage_form':c.get('dosage_form'),
            'strength':c.get('strength'),'manufacturer':c.get('manufacturer'),'indication':c.get('indication'),
            'classification':c.get('classification'),'source':c.get('source'),
            'identity_intelligence':{'identity_completeness':evidence_score,'identity_status':'IDENTIFIED' if name else 'INCOMPLETE',
                                     'public_evidence_status':'REFERENCE_CATALOGUE','source':c.get('source') or 'unknown'},
            'facility_intelligence':facility_state,
            'decision_readiness': 'FACILITY_ANALYTICS_READY' if fac else ('IDENTITY_READY' if evidence_score>=67 else 'IDENTITY_INCOMPLETE'),
            'analysis_boundary':'Public/reference intelligence is available. Facility demand, inventory, forecast, stock-out and reorder conclusions require authorized facility data.'
        })
    return {
      'generated_at':pd.Timestamp.utcnow().isoformat(), 'engine':'StockShield Medicine Intelligence Portfolio v1',
      'scope':{'total_catalogue_identities':total,'returned':len(records),'limit':limit,'offset':offset,'search':search or None},
      'records':records,
      'summary':{
        'total':int(counts['total'] or 0),'identity_completeness_pct':round(sum(int(x['identity_intelligence']['identity_completeness']) for x in records)/len(records),1) if records else 0,
        'with_strength':int(counts['with_strength'] or 0),'with_dosage_form':int(counts['with_form'] or 0),'with_manufacturer':int(counts['with_manufacturer'] or 0),
        'with_category':int(counts['with_category'] or 0),'with_indication':int(counts['with_indication'] or 0),
        'analytics_ready_in_returned':sum(x['facility_intelligence']['status']=='ANALYTICS_READY' for x in records),
        'reference_only_in_returned':sum(x['facility_intelligence']['status']=='REFERENCE_ONLY' for x in records)
      },
      'data_policy':'All catalogue identities are analyzed. No facility stock, sales, forecast, shortage probability or reorder quantity is fabricated for reference-only medicines.'
    }

def system_status():
    meta=ev("meta") or {}
    exists=q("SELECT name FROM sqlite_master WHERE type='table' AND name='medicine_catalogue'")
    cat=q("SELECT COUNT(*) n FROM medicine_catalogue")[0]["n"] if exists else 0
    return {"status":"operational","model_version":core.MODEL_VERSION,"temporal_training_records":meta.get("records",0),"demo_medicines":meta.get("medicines",0),"catalogue_records":cat,"monitored_medicine_identities":cat,"analytics_ready_medicines":meta.get("medicines",0),"catalogue_source":"medicine_dataset.csv" if cat else None,"synthetic_temporal_data":True,"external_pharma_source":"Milan Zdravkovic Pharma Sales Data (public benchmark; external source attribution recorded)","live_public_sources":list(live_sources.SOURCES.keys()),"global_medicine_resolver": {"engine":"GM-RRE v1","sources":list(medicine_resolver.SOURCES.keys()),"search_endpoint":"/api/medicine-search"},"live_policy":"Public live sources adjust planning context only; authorized facility data remains the source of truth."}

def data_quality():
    def build():
        con = sqlite3.connect(DB)
        try: t = [pd.read_sql(f"SELECT * FROM {x}", con) for x in ("medicine", "sales", "inventory")]
        finally: con.close()
        df = core.process(*t); n = len(df); span = sum((g.date.max() - g.date.min()).days + 1 for _, g in df.groupby("medicine_id")); missing = max(0, span - n) / span; d = df.sort_values(["medicine_id", "date"])
        exp = d.groupby("medicine_id").quantity_available.shift(1) + d.quantity_received - d.quantity_sold; mism = int(((exp - d.quantity_available).abs() > 1e-6).sum()); cens = float(df.censored.mean())
        return {"score": int(round(100 * (1 - missing) * (1 - mism / n) * (1 - .5 * cens))), "records": n, "expected_records": span, "missing_dates": span - n, "null_values": int(t[1].isna().sum().sum() + t[2].isna().sum().sum()),
                "inventory_mismatches": mism, "censored_observations": int(df.censored.sum()), "censored_share": cens, "history_start": df.date.min().strftime("%Y-%m-%d"), "history_end": df.date.max().strftime("%Y-%m-%d"),
                "formula": "score = 100 x (1 - missing-date share) x (1 - share of rows where inventory != previous inventory + received - sold) x (1 - 0.5 x censored share)"}
    return cached("dq", build)


def command_center():
    def build():
        ps = [profile(m) for m in q("SELECT * FROM medicine")]; ovs = [p["ov"] for p in ps]; dmax = max(o["forecast_7d"] / 7 for o in ovs) or 1; md = float(np.median([o["forecast_7d"] / 7 for o in ovs])); meta = ev("meta") or {}; rank = []
        for p in ps:
            o = p["ov"]; rank.append({"id": o["id"], "name": o["name"], **intel.priority(o, p["trend"], dmax), "risk_probability": o["risk_probability"], "risk_level": o["risk_level"], "stock": o["stock"], "coverage_days": o["days_remaining"],
                                      "expected_shortage_date": o["expected_stockout_date"], "lead_time_days": o["lead_time_days"], "growth_pct": p["trend"]["growth_pct"], "reorder_required": p["reorder"]["reorder_required"],
                                      "reorder_point": p["reorder"]["reorder_point"], "recommended_reorder": p["reorder"]["recommended_quantity"], "quadrant": intel.quadrant(o, md), "daily_demand": o["forecast_7d"] / 7})
        rank.sort(key=lambda r: -r["score"]); al = intel.tiered_alerts([(p["ov"], p) for p in ps], meta.get("ran_at")); fe = ev("forecast")
        catalogue_count = q("SELECT COUNT(*) n FROM medicine_catalogue")[0]["n"] if q("SELECT name FROM sqlite_master WHERE type='table' AND name='medicine_catalogue'") else 0
        kp = {"monitored": catalogue_count, "analytics_medicines": len(ps), "high_risk": sum(o["risk_level"] == "HIGH" for o in ovs), "critical": sum(r["tier"] == "CRITICAL" for r in rank), "predicted_shortages": sum(bool(o["expected_stockout_date"]) for o in ovs),
              "shortages_24h": sum(o["days_remaining"] <= 1 for o in ovs), "shortages_72h": sum(o["days_remaining"] <= 3 for o in ovs), "unmet_demand_14d": sum(max(0.0, o["forecast_14d"] - o["stock"]) for o in ovs),
              "action_required": sum(r["reorder_required"] or r["risk_level"] == "HIGH" for r in rank), "active_alerts": sum(a["tier"] != "INFORMATION" for a in al), "forecast_mae_7d": fe[0]["rf_mae"],
              "improvement_vs_baseline_pct": 100 * (fe[0]["moving_avg_mae"] - fe[0]["rf_mae"]) / fe[0]["moving_avg_mae"]}
        return {"generated_at": meta.get("ran_at"), "kpis": kp, "ranking": rank, "alerts": al, "reorders": sorted([r for r in rank if r["reorder_required"]], key=lambda r: r["coverage_days"]),
                "balance": [{"id": r["id"], "name": r["name"], "quadrant": r["quadrant"], "coverage_days": r["coverage_days"], "daily_demand": r["daily_demand"]} for r in rank],
                "anomalies": [{"id": p["ov"]["id"], "name": p["ov"]["name"], "status": p["anomalies"]["status"], "items": [i for i in p["anomalies"]["items"] if i["level"] != "NORMAL"]} for p in ps if p["anomalies"]["status"] != "NORMAL"],
                "hidden_demand": sorted([{"id": p["ov"]["id"], "name": p["ov"]["name"], **{k: p["censored"][k] for k in ("observed_daily_14d", "estimated_daily_14d", "censored_days_14d", "censored_days_90d", "stockout_events_90d", "hidden_units_90d", "score", "level")}} for p in ps], key=lambda r: -r["score"]),
                "trends": [{"id": p["ov"]["id"], "name": p["ov"]["name"], **p["trend"]} for p in ps], "data_quality": data_quality()}
    return cached("cc", build)


def alerts():
    ov = {o["id"]: o for o in overview()}; ts = (ev("meta") or {}).get("ran_at"); out = []
    for a in q("SELECT * FROM alert"):
        o = ov[a["medicine_id"]]; out.append({**a, "medicine": o["name"], "timestamp": ts, "stock": o["stock"], "lead_time_days": o["lead_time_days"], "days_remaining": o["days_remaining"], "expected_shortage_date": o["expected_stockout_date"]})
    return {"alerts": sorted(out, key=lambda a: (a["severity"] != "critical", a["days_remaining"] or 1e9)), "generated_at": ts}


def handle(path, qs=None):
    qs = qs or {}; path = path.rstrip("/") or "/"
    try:
        if path == "/health": return 200, {"status": "ok", "analytics_medicines": q("SELECT COUNT(*) n FROM medicine")[0]["n"], "monitored_medicine_identities": q("SELECT COUNT(*) n FROM medicine_catalogue")[0]["n"]}
        if path == "/medicines": return 200, overview()
        if path == "/alerts": return 200, alerts()
        if path == "/alerts/tiered": return 200, {"alerts": command_center()["alerts"], "generated_at": command_center()["generated_at"]}
        if path == "/command-center": return 200, command_center()
        if path == "/metrics": return 200, {"forecast": ev("forecast"), "risk": ev("risk"), "meta": ev("meta"), "data_quality": data_quality(), "system": system_status(), "disclaimer": "Temporal demand/inventory evaluation currently uses synthetic demo data; the local reference catalogue contains thousands of medicine records and live FDA/NLM/CDC context is applied when network access is available."}
        if path == "/system-status": return 200, system_status()
        if path == "/live-status": return 200, live_sources.system_snapshot(force=False)
        if path == "/live-refresh": return 200, live_sources.system_snapshot(force=True)
        if path == "/catalogue": return 200, catalogue(qs.get("search"), qs.get("limit",100), qs.get("offset",0), qs.get("live",False))
        if path == "/monitoring": return 200, monitoring_universe(qs.get("search"), qs.get("limit",500), qs.get("offset",0))
        if path == "/medicine-search": return 200, medicine_search(qs.get("q") or qs.get("search"), qs.get("limit",20), qs.get("web",True), qs.get("force",False))
        if path == "/medicine-intelligence": return 200, medicine_intelligence(qs.get("search"), qs.get("limit",50), qs.get("offset",0), qs.get("sort","priority"))
        if path == "/data-quality": return 200, data_quality()
        m = re.fullmatch(r"/(medicines|forecast|risk|history|simulate|recommendations|anomalies|inventory-health|inventory-projection|censored-demand|live-intelligence)/([^/]+)", path)
        if m:
            row = med(m[2]); kind = m[1]
            if kind == "medicines": return 200, {**row, **profile(row)["ov"], "live_context_available": True}
            if kind == "simulate": return 200, simulate(row, qs)
            if kind == "live-intelligence": return 200, live_intelligence(row)
            pr = lambda k: {"medicine_id": row["id"], **profile(row)[k]}
            return 200, {"forecast": lambda: forecast(row), "risk": lambda: risk(row), "history": lambda: history(row), "recommendations": lambda: {**pr("reorder"), "disclaimer": REORDER_NOTE}, "anomalies": lambda: pr("anomalies"),
                         "inventory-health": lambda: pr("health"), "inventory-projection": lambda: pr("projection"), "censored-demand": lambda: pr("censored")}[kind]()
        raise HttpError(404, "not found")
    except HttpError as e: return e.code, {"error": e.msg}
    except Exception: log.exception("request failed"); return 500, {"error": "internal error"}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/"): code, obj = handle(u.path[4:], {k: v[0] for k, v in parse_qs(u.query).items()}); body, ct = json.dumps(obj).encode(), "application/json"
        else:
            with open(os.path.join(WEB, "index.html"), "rb") as fh: body = fh.read()
            code, ct = 200, "text/html; charset=utf-8"
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a): pass


def main(port=int(os.getenv("PORT", "8000"))):
    if not os.path.exists(DB): core.run(DB)
    print(f"StockShield AI on http://localhost:{port}"); ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__": main()
