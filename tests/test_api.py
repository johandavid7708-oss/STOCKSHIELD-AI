import json, os, tempfile, threading, unittest, urllib.request
from http.server import ThreadingHTTPServer
from stockshield import core, server


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        server.DB = os.path.join(tempfile.mkdtemp(), "api.db"); core.run(server.DB)

    def test_health_and_medicines(self):
        self.assertEqual(server.handle("/health")[1]["status"], "ok"); code, m = server.handle("/medicines"); self.assertEqual((code, len(m)), (200, 6)); self.assertIn(m[0]["risk_level"], ("LOW", "MEDIUM", "HIGH"))
        self.assertEqual(server.handle("/medicines/1")[1]["id"], 1)

    def test_forecast_risk_history(self):
        f = server.handle("/forecast/2")[1]; self.assertEqual(len(f["daily"]), 14); self.assertGreater(f["baseline_7d"], 0)
        r = server.handle("/risk/2")[1]; self.assertTrue(r["drivers"]); self.assertGreaterEqual(r["risk_probability"], 0)
        h = server.handle("/history/2")[1]; self.assertEqual(len(h["days"]), 90); self.assertTrue(any(d["censored"] for i in range(1, 7) for d in server.handle(f"/history/{i}")[1]["days"]), "expected at least one censored day in the seeded data")

    def test_alerts_metrics(self):
        a = server.handle("/alerts")[1]["alerts"]; sev = [x["severity"] for x in a]; self.assertEqual(sev, sorted(sev, key=lambda s: s != "critical"))
        m = server.handle("/metrics")[1]; self.assertIn("synthetic", m["disclaimer"]); self.assertEqual(len(m["forecast"]), 2); self.assertEqual(m["meta"]["medicines"], 6)

    def test_invalid(self):
        for p, c in (("/medicines/999", 404), ("/forecast/abc", 400), ("/risk/0", 404), ("/nope", 404)): self.assertEqual(server.handle(p)[0], c, p)
        self.assertEqual(server.handle("/simulate/1", {"stock": "-5"})[0], 400); self.assertEqual(server.handle("/simulate/1", {"lead": "x"})[0], 400)

    def test_simulate_uses_backend(self):
        lo = server.handle("/simulate/1", {"stock": "10", "lead": "5"})[1]; hi = server.handle("/simulate/1", {"stock": "5000", "lead": "5"})[1]
        self.assertGreater(lo["risk_probability"], hi["risk_probability"]); self.assertLess(lo["coverage_days"], hi["coverage_days"])
        self.assertGreater(server.handle("/simulate/1", {"stock": "10", "lead": "3", "reorder_qty": "5000"})[1]["projected_inventory"][-1], lo["projected_inventory"][-1])


    def test_catalogue_has_thousands_and_supports_live_enrichment(self):
        c = server.handle("/catalogue", {"limit": "5"})[1]
        self.assertGreaterEqual(c["total"], 3000)
        self.assertGreaterEqual(len(c["records"]), 1)
        original = server.live_sources.catalogue_lookup
        try:
            server.live_sources.catalogue_lookup = lambda term, limit=20, force=False: {"query": term, "rxnorm": [{"name": "Amoxicillin", "rxcui": "723", "source": "NLM RxNorm"}], "fda_ndc": [], "sources": {}}
            live = server.handle("/catalogue", {"search": "amoxicillin", "limit": "5", "live": "1"})[1]
            self.assertIn("live_lookup", live); self.assertEqual(live["live_lookup"]["rxnorm"][0]["name"], "Amoxicillin")
        finally:
            server.live_sources.catalogue_lookup = original

    def test_stock_coverage_scales_with_inventory(self):
        small = server.handle("/simulate/1", {"stock": "56", "lead": "5"})[1]
        large = server.handle("/simulate/1", {"stock": "56000", "lead": "5"})[1]
        self.assertGreater(large["coverage_days"], small["coverage_days"] * 100)
        self.assertLess(large["risk_probability"], small["risk_probability"])
        self.assertGreater(large["coverage_days"], 100.0)

    def test_live_overlay_propagates(self):
        original = server.live_sources.build_live_intelligence
        sample = {"medicine_id": 1, "medicine": "Paracetamol 500mg", "live": True, "data_mode": "LIVE_PUBLIC_CONTEXT",
                  "supply": {"shortage_matches": 2, "pressure": .66, "records": []},
                  "safety": {"recall_matches": 1, "pressure": .33},
                  "respiratory_context": {"relevant": False, "pressure": 0, "pathogens": {}},
                  "historical_benchmark": {"available": True, "growth_pct": 20, "direction": "increasing"},
                  "identity": {"rxnorm_available": True, "label_matches": 1}, "provenance": {"sources": {}}}
        try:
            server._cache.clear()
            server.live_sources.build_live_intelligence = lambda m, force=False: sample
            base = server.handle("/forecast/1")[1]
            risk = server.handle("/risk/1")[1]
            self.assertNotEqual(base["forecast_7d"], server.q("SELECT SUM(predicted_demand) s FROM forecast WHERE medicine_id=1")[0]["s"])
            self.assertGreaterEqual(risk["risk_probability"], risk["base_risk_probability"])
            self.assertGreater(base["live_context"]["demand_shift_pct"], 0)
        finally:
            server.live_sources.build_live_intelligence = original
            server._cache.clear()

    def test_global_medicine_resolver_endpoint(self):
        original = server.medicine_resolver.resolve
        try:
            server.medicine_resolver.resolve = lambda q, limit=20, force=False, web=True: {
                "query": q, "engine": "GM-RRE v1", "results": [{"name": "Rifaximin 550 MG Oral Tablet", "sources": ["NLM RxNorm", "FDA NDC", "DailyMed"], "evidence_count": 3, "match_score": 0.99, "evidence": [{"rxcui": "123"}]}],
                "web_discovery": [], "source_status": {"rxnorm": {"ok": True, "cached": False, "error": None}}
            }
            code, out = server.handle("/medicine-search", {"q": "rifaximin", "limit": "20", "web": "1"})
            self.assertEqual(code, 200); self.assertEqual(out["engine"], "GM-RRE v1"); self.assertEqual(out["results"][0]["name"], "Rifaximin 550 MG Oral Tablet"); self.assertGreaterEqual(out["results"][0]["match_score"], .9)
        finally:
            server.medicine_resolver.resolve = original

    def test_http_integration(self):
        s = ThreadingHTTPServer(("127.0.0.1", 0), server.H); threading.Thread(target=s.serve_forever, daemon=True).start(); b = f"http://127.0.0.1:{s.server_port}"
        self.assertEqual(json.load(urllib.request.urlopen(b + "/api/health"))["status"], "ok"); self.assertIn(b"StockShield", urllib.request.urlopen(b + "/").read())
        try: urllib.request.urlopen(b + "/api/medicines/999")
        except urllib.error.HTTPError as e: self.assertEqual(e.code, 404)
        s.shutdown()


if __name__ == "__main__": unittest.main()


class Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(c): server.DB = os.path.join(tempfile.mkdtemp(), "c.db"); core.run(server.DB)

    def test_math_consistency(self):
        for m in server.handle("/medicines")[1]:
            i = m["id"]; f = server.handle(f"/forecast/{i}")[1]; d = [x["predicted_demand"] for x in f["daily"]]; r = server.handle(f"/risk/{i}")[1]
            self.assertAlmostEqual(f["forecast_7d"], sum(d[:7])); self.assertAlmostEqual(f["forecast_14d"], sum(d)); self.assertAlmostEqual(m["forecast_7d"], f["forecast_7d"])
            self.assertAlmostEqual(f["baseline_14d"], 2 * f["baseline_7d"]); self.assertTrue(all(x["lower_bound"] <= x["predicted_demand"] <= x["upper_bound"] for x in f["daily"]))
            s = server.handle(f"/simulate/{i}", {"stock": str(m["stock"]), "lead": str(m["lead_time_days"])})[1]
            self.assertAlmostEqual(s["coverage_days"], r["days_remaining"], places=6); self.assertAlmostEqual(s["risk_probability"], r["risk_probability"], places=9)
            neg = [j for j, v in enumerate(s["projected_inventory"], 1) if v <= 0]; self.assertEqual(s["shortage_day"], neg[0] if neg else None)
            self.assertIn(f"{m['stock']:.0f} units", r["explanation"]); self.assertIn(f"{m['forecast_7d']:.0f} units", r["explanation"])
            ps = [server.handle(f"/simulate/{i}", {"stock": str(x)})[1]["risk_probability"] for x in (0, 50, 200, 1000, 5000)]; self.assertEqual(ps, sorted(ps, reverse=True))

    def test_bad_inputs(self):
        for qs in ({"stock": "nan"}, {"stock": "inf"}, {"stock": "-1"}, {"lead": "0"}, {"lead": "61"}, {"reorder_qty": "-1"}, {"stock": "abc"}): self.assertEqual(server.handle("/simulate/1", qs)[0], 400, qs)
        for p in ("/medicines/1.5", "/medicines/-1", "/medicines/%20", "/medicines/²", "/medicines/99999999999999999999", "/forecast/", "/risk/1/x"): self.assertIn(server.handle(p)[0], (400, 404), p)
