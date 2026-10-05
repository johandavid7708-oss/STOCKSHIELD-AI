"""Deterministic decision-intelligence layer: trend, hidden demand, projection, reorder, health, anomalies, priority, tiered alerts.
Everything is computed from stored data and forecasts; nothing here is a trained model or a hardcoded result."""
import math
import pandas as pd

TIERS = ["CRITICAL", "HIGH", "WATCH", "INFORMATION"]


def _f(x): return 0.0 if pd.isna(x) else float(x)


def trend(d):
    """d = stock-out-corrected daily demand. Recent = last 7 days; baseline = the 28 days before that; z uses the baseline's daily sd (standard error of a 7-day mean)."""
    r, prev = d.tail(7).mean(), d.iloc[-35:-7]; b = prev.mean(); sd = _f(prev.std()); g = (r - b) / b if b and b > 0 else 0.0; z = (r - b) / (sd / math.sqrt(7)) if sd > 0 else 0.0
    return dict(recent_daily=_f(r), baseline_daily=_f(b), growth_pct=_f(g * 100), direction="INCREASING" if g > .1 else "DECREASING" if g < -.1 else "STABLE",
                volatility_cv=_f(d.tail(28).std()) / max(_f(d.tail(28).mean()), 1e-9), acceleration=_f(r - d.iloc[-14:-7].mean()), z_score=_f(z),
                spike_level="EXTREME SPIKE" if z >= 5 else "SPIKE" if z >= 3 else "ELEVATED" if z >= 1.5 else "NORMAL")


def censored(df):
    w = df.tail(90); c = w.censored.astype(bool); ev = int((c & ~c.shift(fill_value=False)).sum()); share = float(c.mean()); r = df.tail(14)
    hidden = _f((w.demand_adj - w.quantity_sold)[c].sum()); daily = _f(w.demand_adj[~c].mean()); score = 40 * min(1, share / .15) + 30 * min(1, ev / 4) + 30 * min(1, hidden / max(daily * 7, 1e-9))
    return dict(observed_daily_14d=_f(r.quantity_sold.mean()), estimated_daily_14d=_f(r.demand_adj.mean()), censored_days_14d=int(r.censored.sum()), censored_days_90d=int(c.sum()), stockout_events_90d=ev,
                current_inventory=_f(df.quantity_available.iloc[-1]), hidden_units_90d=hidden, uncensored_daily_90d=daily, score=int(round(score)), level="HIGH" if score >= 60 else "MODERATE" if score >= 30 else "LOW",
                formula="score = 40*min(1, censored share of last 90 days / 15%) + 30*min(1, stock-out events in last 90 days / 4) + 30*min(1, hidden units / one week of uncensored demand). Deterministic heuristic; not clinically validated.")


def projection(stock, safety, fc, lead):
    import numpy as np
    t = stock - np.cumsum(fc); first = lambda cond: next((i + 1 for i, v in enumerate(t) if cond(v)), None)
    return dict(current_stock=float(stock), safety_stock=float(safety), lead_time_days=int(lead), projected=[round(float(v), 1) for v in t], safety_breach_day=first(lambda v: v < safety),
                shortage_day=first(lambda v: v <= 0), replenishment_assumed=False, note="No replenishment is assumed: stored data holds no open purchase orders. Use the scenario lab to model a reorder.")


def reorder(stock, safety, daily, lead, f7, growth=0.0):
    ltd = daily * lead; rop = ltd + safety; up = rop + f7; q = max(0, math.ceil(up - stock)); need = stock <= rop
    why = [f"Current stock {stock:.0f} units is {'at or below' if need else 'above'} the reorder point of {rop:.0f}", f"Expected demand during the {lead}-day lead time is {ltd:.0f} units ({'exceeds' if ltd > stock else 'is within'} current stock)",
           f"Safety stock requirement: {safety:.0f} units"] + ([f"Recent demand trend is {growth:+.0f}%"] if abs(growth) >= 10 else []) + ([f"An order of {q} units restores stock to {up:.0f} (reorder point + one week of forecast demand)"] if need else [])
    return dict(current_stock=float(stock), lead_time_days=int(lead), expected_lead_time_demand=ltd, safety_stock=float(safety), reorder_point=rop, order_up_to_level=up, recommended_quantity=int(q), reorder_required=bool(need), why=why,
                policy="reorder point = forecast daily demand x lead time + safety stock; order-up-to level = reorder point + 7-day forecast demand; recommended quantity = max(0, order-up-to level - current stock)")


def health(p, cov, lead, tr, f7, sigma7, cens_share):
    ceil = 2 * (lead + 7); cf = cov / lead if cov < lead else 1.0 if cov <= ceil else max(0.0, 1 - (cov - ceil) / ceil)
    parts = {"shortage_risk": (35, 1 - p), "coverage": (20, cf), "demand_stability": (15, 1 - min(1, tr["volatility_cv"])), "forecast_confidence": (15, 1 - min(1, sigma7 / max(f7, 1e-9))), "replenishment_reliability": (15, 1 - min(1, cens_share / .2))}
    comp = {k: round(w * max(0.0, min(1.0, v)), 1) for k, (w, v) in parts.items()}; s = int(round(sum(comp.values())))
    return dict(score=s, label="GOOD" if s >= 70 else "FAIR" if s >= 45 else "POOR", components=comp, max_points={k: w for k, (w, _) in parts.items()}, overstock_risk=bool(cov > ceil),
                formula="100 points = shortage risk (35, 1-probability) + coverage (20, full when coverage lies between lead time and 2x(lead time+7 days)) + demand stability (15, 1-CV of last 28 days) + forecast confidence (15, 1-validation sd/7-day forecast) + replenishment reliability (15, 1-censored share of last 90 days/20%)")


def anomalies(df, tr, ce):
    it = []; add = lambda lv, t, msg, why: it.append(dict(level=lv, type=t, message=msg, why=why)); z = tr["z_score"]
    if z >= 1.5: add("ANOMALY" if z >= 3 else "WATCH", "DEMAND_SPIKE", f"Demand is {z:.1f}σ above its recent baseline", [f"Recent demand {tr['recent_daily']:.1f}/day vs baseline {tr['baseline_daily']:.1f}/day ({tr['growth_pct']:+.0f}%)", "Measured on stock-out-corrected demand against the previous 28-day distribution"])
    u = df.tail(7); u = u[~u.censored]; bu = df.iloc[-35:-7]; bu = bu[~bu.censored].demand_adj
    if len(u) >= 3 and len(bu) >= 10 and bu.std() > 0:
        zd = (u.demand_adj.mean() - bu.mean()) / (bu.std() / math.sqrt(len(u)))
        if zd <= -1.5: add("ANOMALY" if zd <= -3 else "WATCH", "SUDDEN_DROP", f"Demand is {abs(zd):.1f}σ below its recent baseline", [f"{len(u)} uncensored recent days average {u.demand_adj.mean():.1f}/day vs baseline {bu.mean():.1f}/day", "Inventory remained available during these observations, so the drop is not explained by a stock-out"])
    if ce["censored_days_14d"] > 0: add("NORMAL", "STOCKOUT_ZERO_SALES_EXPLAINED", f"{ce['censored_days_14d']} recent low-sales day(s) coincide with zero inventory", ["Low observed sales are attributed to unavailability, not to a demand collapse", f"Demand estimated at {ce['estimated_daily_14d']:.1f}/day vs {ce['observed_daily_14d']:.1f}/day observed"])
    t30 = df.tail(30); zs = int(((t30.quantity_sold == 0) & ~t30.censored).sum())
    if zs >= 2: add("ANOMALY" if zs >= 4 else "WATCH", "UNEXPECTED_ZERO_SALES", f"{zs} zero-sales day(s) in the last 30 while stock was available", ["Inventory was positive on these days, so zero sales are not explained by a stock-out"])
    if ce["stockout_events_90d"] >= 3: add("ANOMALY" if ce["stockout_events_90d"] >= 5 else "WATCH", "REPEATED_STOCKOUTS", f"{ce['stockout_events_90d']} stock-out events in the last 90 days", [f"{ce['censored_days_90d']} censored days in the last 90", "Repeated stock-outs point to a replenishment problem"])
    d = df.demand_adj; prev = d.iloc[-84:-28]; sd = _f(prev.std()); zr = (d.tail(28).mean() - prev.mean()) / (sd / math.sqrt(28)) if sd > 0 else 0.0
    if abs(zr) >= 4: add("ANOMALY" if abs(zr) >= 6 else "WATCH", "DEMAND_REGIME_SHIFT", f"28-day demand level moved {zr:+.1f}σ versus the prior 56 days", [f"Last 28 days average {d.tail(28).mean():.1f}/day vs {prev.mean():.1f}/day before", "May reflect seasonality or a lasting change"])
    lv = [i["level"] for i in it]; return dict(status="ANOMALY" if "ANOMALY" in lv else "WATCH" if "WATCH" in lv else "NORMAL", items=it)


def priority(o, tr, dmax):
    hz = o["lead_time_days"] + 7; daily = o["forecast_7d"] / 7; sf = o["safety_stock"]
    parts = {"stock_out_probability": (40, o["risk_probability"]), "shortage_urgency": (25, (hz - o["days_remaining"]) / hz), "demand_magnitude": (15, daily / max(dmax, 1e-9)), "safety_stock_gap": (10, (sf - o["stock"]) / sf if sf else 0), "demand_trend": (10, tr["growth_pct"] / 50)}
    comp = {k: round(w * max(0.0, min(1.0, v)), 1) for k, (w, v) in parts.items()}; sc = round(sum(comp.values()), 1)
    why = [f"Stock-out probability {o['risk_probability']:.0%}", f"Coverage {o['days_remaining']:.1f} days vs lead time {o['lead_time_days']} days", f"Demand trend {tr['growth_pct']:+.0f}%", f"Forecast demand {daily:.1f} units/day", f"Stock {o['stock']:.0f} vs safety stock {sf:.0f}"]
    return dict(score=sc, tier="CRITICAL" if sc >= 60 else "HIGH" if sc >= 40 else "WATCH" if sc >= 20 else "ROUTINE", components=comp, why=why)


def quadrant(o, median_daily):
    hs = o["days_remaining"] > 2 * (o["lead_time_days"] + 7); hd = o["forecast_7d"] / 7 >= median_daily
    return ("HEALTHY" if hd else "OVERSTOCK") if hs else ("SHORTAGE RISK" if hd else "WATCH")


def tiered_alerts(items, ts):
    out = []
    for ov, pr in items:
        d, lead, p, ro, tr = ov["days_remaining"], ov["lead_time_days"], ov["risk_probability"], pr["reorder"], pr["trend"]; n0 = len(out); nm = ov["name"]
        m = dict(stock=ov["stock"], coverage_days=round(d, 1), lead_time_days=lead, stock_out_probability=round(p, 3), forecast_7d=round(ov["forecast_7d"], 1))
        act = f"Order about {ro['recommended_quantity']} units now; supplier lead time is {lead} days." if ro["reorder_required"] else "Monitor; stock is above the reorder point."
        add = lambda tier, title, reason, timing, action=act: out.append(dict(tier=tier, medicine_id=ov["id"], medicine=nm, title=title, reason=reason, timing=timing, action=action, metrics=m, timestamp=ts))
        date = ov["expected_stockout_date"] or "beyond 14 days"
        if d <= 1: add("CRITICAL", f"{nm} may stock out within 24 hours", f"Coverage is about {d * 24:.0f} hours of forecast demand.", date)
        elif p >= .8 and d < lead: add("CRITICAL", f"{nm}: very likely shortage before replenishment arrives", f"Stock-out probability {p:.0%}; coverage {d:.1f} days vs lead time {lead} days.", date)
        elif ov["risk_level"] == "HIGH" or d < lead: add("HIGH", f"{nm} projected to run short before replenishment can arrive", f"Coverage {d:.1f} days vs lead time {lead} days; stock-out probability {p:.0%}.", date)
        elif pr["projection"]["safety_breach_day"] and pr["projection"]["safety_breach_day"] <= 2: add("HIGH", f"{nm} projected to breach safety stock within 48 hours", f"Projected inventory falls below the safety stock of {ov['safety_stock']} on day {pr['projection']['safety_breach_day']}.", date)
        elif ov["stock"] < ov["safety_stock"]: add("WATCH", f"{nm} is below safety stock", f"Stock {ov['stock']:.0f} vs safety stock {ov['safety_stock']}.", date)
        if tr["growth_pct"] >= 20 or tr["spike_level"] != "NORMAL": add("WATCH", f"{nm}: demand increased {tr['growth_pct']:+.0f}% over baseline", f"Recent demand {tr['recent_daily']:.1f}/day vs {tr['baseline_daily']:.1f}/day ({tr['spike_level']}).", "ongoing", "Review the forecast and consider ordering ahead of the lead time.")
        if len(out) == n0 and ro["reorder_required"]: add("INFORMATION", f"Reorder recommended for {nm}", f"Stock {ov['stock']:.0f} is at or below the reorder point {ro['reorder_point']:.0f}.", date)
        if len(out) == n0 and pr["censored"]["censored_days_14d"] > 0 and ov["stock"] > 0: add("INFORMATION", f"{nm} recovered from a recent stock-out", f"{pr['censored']['censored_days_14d']} censored day(s) in the last 14; demand was corrected for them.", "recent", "No action needed; forecasts already use corrected demand.")
    return sorted(out, key=lambda a: (TIERS.index(a["tier"]), a["metrics"]["coverage_days"]))
