# StockShield AI — Pharmaceutical Inventory Command Center

Detects demand hidden by stock-outs, forecasts 7/14-day demand, predicts shortages, explains the risk, recommends reorders, raises tiered alerts and lets you test scenarios. Inventory decision support only — not medical advice.

## Run (Python 3.10+; Node only for the optional UI check)
    pip install -r requirements.txt        # pandas, numpy, scikit-learn
    python -m stockshield.server           # first start seeds SQLite, trains models, serves
    # open http://localhost:8000
Re-seed/retrain only: `python -m stockshield.core [db]` (replaces table contents). Env: `STOCKSHIELD_DB`, `PORT`.
Tests: `python -m unittest discover -v` (23 tests). UI logic check with server running: `node scripts/ui_smoke.js` (11 checks, runs the page JavaScript in Node against the live API; it is not a browser).

## Architecture
SQLite -> `stockshield/core.py` (data, censoring, features, Random Forest, risk, alerts) -> `stockshield/intel.py` (deterministic decision layer) -> `stockshield/server.py` (stdlib HTTP, JSON under `/api`, cached per database state) -> `stockshield/web/index.html` (vanilla JS + inline SVG). No React, FastAPI, XGBoost or Docker.

## Core ML (unchanged)
Censored days (end-of-day stock = 0) get demand = max(sales, trailing 28-day uncensored mean), past data only; raw sales kept. Random Forest forecasts 7- and 14-day demand on lag/rolling/trend/calendar/inventory/lead-time features, validated temporally against a moving-average baseline. Risk = P(demand over lead time > stock - safety stock) (normal approximation); HIGH >= 0.6, MEDIUM >= 0.3. Uncertainty band = approximate validation-residual band, not a formal confidence interval.

## Decision layer (`intel.py`, all computed from stored data)
- **Trend:** last 7 days vs previous 28; growth %, direction (±10%), volatility (CV), acceleration; spike level from z-score (NORMAL <1.5, ELEVATED, SPIKE >=3, EXTREME >=5).
- **Censored-demand score (0–100):** 40×min(1, censored share of 90d/15%) + 30×min(1, stock-out events/4) + 30×min(1, hidden units/one week of uncensored demand).
- **Projection:** stock minus cumulative forecast; safety-breach and shortage days. No replenishment is assumed (no purchase orders are stored).
- **Reorder:** reorder point = forecast daily demand × lead time + safety stock; order-up-to = reorder point + 7-day forecast; quantity = max(0, order-up-to − stock). Deterministic, not guaranteed optimal.
- **Inventory health (0–100):** shortage risk 35, coverage 20, demand stability 15, forecast confidence 15, replenishment reliability 15; overstock flag when coverage > 2×(lead time+7 days).
- **Anomalies:** demand spike, sudden drop (uncensored days only), unexpected zero sales while stocked, repeated stock-outs, regime shift. Zero sales during a stock-out are explained as censoring, not a demand collapse.
- **Priority score:** probability 40, shortage urgency 25, demand size 15, safety-stock gap 10, trend 10.
- **Balance matrix:** stock high if coverage > 2×(lead+7) days; demand high if at/above portfolio median.
- **Tiered alerts:** CRITICAL / HIGH / WATCH / INFORMATION, each with reason, timing, action and metrics.
- **30-day figure:** the 14-day rate extrapolated and labelled an extended planning estimate; no 30-day model is validated.

## API (GET, JSON)
`/api/health, medicines, medicines/{id}, forecast/{id}, risk/{id}, history/{id}, alerts, alerts/tiered, metrics, command-center, recommendations/{id}, anomalies/{id}, inventory-health/{id}, inventory-projection/{id}, censored-demand/{id}, simulate/{id}?stock=&lead=&reorder_qty=&demand_mult=&safety=`. Bad IDs/parameters return 400/404 JSON; unexpected errors return a generic 500 and are logged.

## Dashboard
Executive view (KPIs, hidden-demand explainer, reorders, trends, balance matrix, anomalies, filterable table), Stock-out command center (ranked), medicine dossier (overview, why, censored demand, trend, forecast vs baseline, projection, reorder, anomalies, health, scenario lab with 5 presets), Alerts, Model & data (metrics, data trust, data quality), Judge demo (guided links).

## Limitations
All data and metrics are synthetic (optimistic risk scores). No real-data importer; single location; models are retrained each run, not saved. Forecast KPI is MAE, not an accuracy percentage. The UI has never been viewed in a real browser (none was available); layout and responsiveness are unverified. On Windows, security policies (e.g. Smart App Control) may block scikit-learn.

## Round 3 — Real-world public intelligence layer

This build adds `stockshield/live_sources.py`, which connects to public sources for contextual intelligence:

- FDA openFDA Drug Shortages — current national shortage records
- FDA openFDA Drug Enforcement — recall/enforcement context
- FDA openFDA Drug Labeling — product identity/label context
- NIH/NLM RxNorm — standardized medicine identity
- CDC public respiratory surveillance — recent and historical respiratory trend context
- Public Pharma Sales Time Series benchmark — 2014–2019 historical ATC-category demand context

The public data layer is deliberately **not** treated as a replacement for facility transaction data. Public APIs do not expose a pharmacy's current units sold, current inventory, purchase orders, or supplier lead times. Those remain facility-authorized inputs.

New API routes:

- `/api/live-status`
- `/api/live-refresh`
- `/api/live-intelligence/{medicine_id}`

The system caches public responses and displays whether a source is live, cached, or unavailable. It never fabricates a live value when a source cannot be reached.

--- ROUND 3 LIVE DATA ---
See LIVE_DATA.md and README_ROUND3_LIVE.md for the live-source architecture, provenance, cache behavior, and accuracy boundaries.

--- GLOBAL MEDICINE RESOLVER ---
See `GLOBAL_MEDICINE_RESOLVER.md`. The embedded GM-RRE search layer resolves arbitrary medicine/product queries through NLM RxNorm, FDA NDC/labels, DailyMed and PubChem, with optional web-index discovery. Use `/api/medicine-search?q=<medicine>` or the Medicine Search UI.

--- MEDICINE MONITORING UNIVERSE ---

StockShield now separates medicine identity coverage from facility analytics coverage.

- Local monitored medicine identity universe: 3,972 records in the current build.
- Analytics-ready facility medicines: 6, because only those medicines currently have local inventory/sales history.
- Reference-only medicines do not receive fabricated stock, demand, risk, or forecast values.
- API: `GET /api/monitoring?search=<term>&limit=<n>&offset=<n>`
- Global Medicine Search can resolve additional medicines from public registries and cache their identity evidence into the reference universe.

## Portfolio Medicine Intelligence (R10)

The Medicine Intelligence layer now covers the complete local reference catalogue (3,972 identities) through a server-side, paginated endpoint:

- `GET /api/medicine-intelligence?limit=50&offset=0`
- optional `search=` across name, category, manufacturer, strength and dosage form
- identity completeness and provenance for every catalogue identity
- explicit `REFERENCE_ONLY` vs `ANALYTICS_READY` facility-data boundary
- no fabricated stock, sales, forecast, shortage probability or reorder quantity for catalogue-only identities
- frontend route `#/intelligence` exposes the complete portfolio without loading all 3,972 records into the browser at once

Facility-level forecast/risk/reorder calculations remain available through the existing medicine profile APIs when authorized facility telemetry exists.
