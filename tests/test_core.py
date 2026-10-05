import os, sqlite3, tempfile, unittest
import numpy as np, pandas as pd
from stockshield import core


def toy():
    dates = pd.date_range("2025-01-01", periods=45).strftime("%Y-%m-%d"); meds = pd.DataFrame([(1, "X", "c", "u", "s", 5, 20)], columns=["id", "name", "category", "unit", "supplier", "lead_time_days", "safety_stock"])
    sold = np.full(45, 10.0); avail = np.full(45, 100.0); sold[30:34] = 0; avail[30:34] = 0
    return meds, pd.DataFrame({"medicine_id": 1, "date": dates, "quantity_sold": sold}), pd.DataFrame({"medicine_id": 1, "date": dates, "quantity_available": avail, "quantity_received": 0.0})


class Core(unittest.TestCase):
    def test_stockout_detection(self):
        ev = core.detect_stockouts(core.process(*toy())); self.assertEqual(len(ev), 1); self.assertEqual(int(ev.duration[0]), 4)

    def test_censored_demand(self):
        df = core.process(*toy()); c = df[df.censored]
        self.assertTrue((c.quantity_sold == 0).all()); self.assertTrue((c.demand_adj > 5).all()); self.assertTrue((df[~df.censored].demand_adj == df[~df.censored].quantity_sold).all())

    def test_no_future_leakage(self):
        df = core.process(*core.synth(days=120)); T = pd.Timestamp("2025-03-15"); a = core.features(df, 7)
        df2 = df.copy(); df2.loc[df2.date > T, "demand_adj"] = 9999; b = core.features(df2, 7)
        pd.testing.assert_frame_equal(a[a.date <= T][core.FEATS].reset_index(drop=True), b[b.date <= T][core.FEATS].reset_index(drop=True))

    def test_shortage_date(self):
        d, k = core.shortage(25, np.full(14, 10.0)); self.assertEqual(k, 3); self.assertAlmostEqual(d, 2.5)
        d, k = core.shortage(1000, np.full(14, 10.0)); self.assertIsNone(k); self.assertGreater(d, 14)

    def test_risk_monotonic(self):
        lo, hi = core.shortfall_prob(500, 20, 5, 10, 15), core.shortfall_prob(60, 20, 5, 10, 15); self.assertLess(lo, hi); self.assertEqual(core.level(.7), "HIGH"); self.assertEqual(core.level(.1), "LOW")

    def test_validation_rejects_bad_data(self):
        m, s, i = toy(); s.loc[0, "quantity_sold"] = -1
        with self.assertRaises(ValueError): core.validate(m, s, i)

    def test_pipeline_end_to_end(self):
        db = os.path.join(tempfile.mkdtemp(), "t.db"); r = core.run(db); con = sqlite3.connect(db)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0], r["medicines"] * 14); self.assertEqual(con.execute("SELECT COUNT(*) FROM risk_prediction").fetchone()[0], r["medicines"])
        for e in r["forecast_eval"]: self.assertTrue(np.isfinite(e["rf_mae"]) and np.isfinite(e["moving_avg_mae"]))
        self.assertGreater(r["censored_days"], 0)


if __name__ == "__main__": unittest.main()
