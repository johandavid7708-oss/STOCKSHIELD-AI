"""Transparent live-context overlay for StockShield.

This is deliberately NOT presented as a replacement for the trained demand model.
It is a bounded evidence layer that propagates fresh public signals into planning
outputs while keeping the base model and provenance visible.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from . import core, intel

MAX_DEMAND_SHIFT = 0.15


def _clip(v, lo, hi):
    return max(lo, min(hi, float(v)))


def build_overlay(live, category="", name=""):
    bench = live.get("historical_benchmark") or {}
    growth = float(bench.get("growth_pct") or 0.0) / 100.0 if bench.get("available") else 0.0
    respiratory = live.get("respiratory_context") or {}
    resp = float(respiratory.get("pressure") or 0.0) if respiratory.get("relevant") else 0.0
    shortage = live.get("supply") or {}
    recall = live.get("safety") or {}

    # Demand context is intentionally conservative: historical public benchmark trend
    # contributes more than a current epidemiology signal, and the total is bounded.
    benchmark_component = _clip(growth * 0.35, -0.10, 0.10)
    respiratory_component = _clip(resp * 0.40, 0.0, 0.05)
    demand_shift = _clip(benchmark_component + respiratory_component, -MAX_DEMAND_SHIFT, MAX_DEMAND_SHIFT)

    # Supply/safety signals do not pretend to change patient demand. They increase
    # operational risk because a national shortage/recall can make replenishment less reliable.
    shortage_matches = int(shortage.get("shortage_matches") or 0)
    recall_matches = int(recall.get("recall_matches") or 0)
    shortage_bump = _clip(0.06 * min(shortage_matches, 3), 0.0, 0.18)
    recall_bump = _clip(0.03 * min(recall_matches, 3), 0.0, 0.09)
    risk_bump = _clip(shortage_bump + recall_bump, 0.0, 0.25)

    return {
        "enabled": bool(live.get("live")),
        "demand_multiplier": round(1.0 + demand_shift, 4),
        "demand_shift_pct": round(demand_shift * 100, 2),
        "risk_probability_bump": round(risk_bump, 4),
        "components": {
            "public_benchmark_growth_pct": round(growth * 100, 2),
            "benchmark_weight": 0.35,
            "respiratory_pressure": round(resp, 4),
            "respiratory_weight": 0.40,
            "shortage_matches": shortage_matches,
            "shortage_risk_bump": round(shortage_bump, 4),
            "recall_matches": recall_matches,
            "recall_risk_bump": round(recall_bump, 4),
        },
        "policy": "Bounded public-signal overlay. It adjusts planning context, not clinical decisions; authorized facility transactions remain the source of truth.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def apply_forecast(base_fc, live, category="", name=""):
    overlay = build_overlay(live, category, name)
    mult = overlay["demand_multiplier"]
    adjusted = []
    for row in base_fc:
        if isinstance(row, dict):
            p = float(row["predicted_demand"])
            lo = float(row.get("lower_bound", p))
            hi = float(row.get("upper_bound", p))
            adjusted.append({**row, "base_predicted_demand": p,
                             "predicted_demand": round(p * mult, 3),
                             "lower_bound": round(max(0, lo * mult), 3),
                             "upper_bound": round(max(0, hi * mult), 3),
                             "live_context_adjusted": overlay["enabled"]})
        else:
            adjusted.append(float(row) * mult)
    return adjusted, overlay


def apply_risk(probability, overlay):
    return _clip(float(probability) + float(overlay.get("risk_probability_bump") or 0), 0.0, 0.999)
