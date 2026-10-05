import os, tempfile, unittest
import numpy as np, pandas as pd
from stockshield import core, intel, server


def make(days=120, censored=(), last7=None):
    dates = pd.date_range("2025-01-01", periods=days).strftime("%Y-%m-%d"); sold = (10 + np.tile([2, -2], days // 2 + 1)[:days]).astype(float); avail = np.full(days, 100.0)
    for i in censored: sold[i] = 0; avail[i] = 0
    if last7 is not None: sold[-7:] = last7
    meds = pd.DataFrame([(1, "X", "c", "u", "s", 5, 20)], columns=["id", "name", "category", "unit", "supplier", "lead_time_days", "safety_stock"])
    return core.process(meds, pd.DataFrame({"medicine_id": 1, "date": dates, "quantity_sold": sold}), pd.DataFrame({"medicine_id": 1, "date": dates, "quantity_available": avail, "quantity_received": 0.0}))


class Intel(unittest.TestCase):
    def test_hidden_demand_estimate(self):
        df = make(censored=range(113, 120)); c = intel.censored(df)
        self.assertEqual(c["censored_days_14d"], 7); self.assertGreater(c["estimated_daily_14d"], c["observed_daily_14d"]); self.assertGreater(c["hidden_units_90d"], 0); self.assertTrue(0 <= c["score"] <= 100)
        self.assertEqual(intel.censored(make())["score"], 0)

    def test_reorder_and_edges(self):
        r = intel.reorder(50, 20, 10, 5, 70); self.assertEqual((r["reorder_point"], r["recommended_quantity"], r["reorder_required"]), (70, 90, True))
        r = intel.reorder(1000, 20, 10, 5, 70); self.assertEqual((r["recommended_quantity"], r["reorder_required"]), (0, False)); self.assertEqual(intel.reorder(0, 20, 0, 5, 0)["reorder_point"], 20)

    def test_projection(self):
        p = intel.projection(25, 10, [10] * 14, 5); self.assertEqual((p["safety_breach_day"], p["shortage_day"]), (2, 3)); self.assertIsNone(intel.projection(5000, 10, [10] * 14, 5)["shortage_day"])

    def test_anomaly_normal_and_stockout_zero(self):
        self.assertEqual(intel.anomalies(df := make(), t := intel.trend(df.demand_adj), intel.censored(df))["status"], "NORMAL")
        df = make(censored=range(113, 120)); t = intel.trend(df.demand_adj); a = intel.anomalies(df, t, intel.censored(df)); types = [i["type"] for i in a["items"]]
        self.assertNotIn("SUDDEN_DROP", types); self.assertIn("STOCKOUT_ZERO_SALES_EXPLAINED", types); self.assertNotEqual(a["status"], "ANOMALY"); self.assertGreater(df.demand_adj.tail(7).mean(), 8)

    def test_genuine_drop_is_flagged(self):
        df = make(last7=2.0); a = intel.anomalies(df, intel.trend(df.demand_adj), intel.censored(df)); self.assertIn("SUDDEN_DROP", [i["type"] for i in a["items"]]); self.assertEqual(a["status"], "ANOMALY")


class IntelApi(unittest.TestCase):
    @classmethod
    def setUpClass(c): server.DB = os.path.join(tempfile.mkdtemp(), "i.db"); core.run(server.DB)

    def test_new_endpoints_and_errors(self):
        for ep in ("recommendations", "anomalies", "inventory-health", "inventory-projection", "censored-demand"):
            self.assertEqual(server.handle(f"/{ep}/1")[0], 200, ep); self.assertEqual(server.handle(f"/{ep}/999")[0], 404); self.assertEqual(server.handle(f"/{ep}/x")[0], 400)
        f = server.handle("/forecast/1")[1]; self.assertTrue(all(d["predicted_demand"] >= 0 for d in f["daily"])); self.assertIn("Extended planning estimate", f["extended_30d"]["label"]); self.assertEqual(len(f["comparison"]), 2)
        h = server.handle("/inventory-health/1")[1]; self.assertTrue(0 <= h["score"] <= 100); self.assertAlmostEqual(sum(h["components"].values()), h["score"], delta=1)

    def test_scenarios(self):
        s = lambda **k: server.handle("/simulate/1", {a: str(b) for a, b in k.items()})[1]; base = s(stock=500, lead=5)
        self.assertGreater(s(stock=500, lead=5, demand_mult=2)["risk_probability"], base["risk_probability"]); self.assertGreater(s(stock=500, lead=20)["risk_probability"], base["risk_probability"])
        self.assertLess(s(stock=2000, lead=5)["risk_probability"], base["risk_probability"]); self.assertGreater(s(stock=500, lead=5, safety=400)["risk_probability"], base["risk_probability"])
        self.assertIn("current", base); self.assertEqual(server.handle("/simulate/1", {"demand_mult": "0"})[0], 400); self.assertEqual(server.handle("/simulate/1", {"safety": "-1"})[0], 400)

    def test_command_center_consistency(self):
        cc = server.handle("/command-center")[1]; ov = server.handle("/medicines")[1]; k = cc["kpis"]
        self.assertEqual(k["monitored"], server.handle("/monitoring", {"limit": "1"})[1]["total"]); self.assertEqual(k["high_risk"], sum(o["risk_level"] == "HIGH" for o in ov)); sc = [r["score"] for r in cc["ranking"]]; self.assertEqual(sc, sorted(sc, reverse=True))
        order = [a["tier"] for a in server.handle("/alerts/tiered")[1]["alerts"]]; self.assertEqual(order, sorted(order, key=intel.TIERS.index)); self.assertTrue(all(a["action"] and a["reason"] for a in cc["alerts"]))
        self.assertTrue(0 <= cc["data_quality"]["score"] <= 100); self.assertEqual(cc["data_quality"]["inventory_mismatches"], 0); self.assertEqual({r["quadrant"] for r in cc["balance"]} - {"HEALTHY", "OVERSTOCK", "WATCH", "SHORTAGE RISK"}, set())


if __name__ == "__main__": unittest.main()
