# StockShield AI — Live Real-World Prototype · Round 4

This build upgrades Round 2 from a UI prototype into a real-world-ready intelligence prototype.

## Start

```bash
pip install -r requirements.txt
python -m stockshield.server
```

Open `http://localhost:8000`.

## Live public data

The app can fetch current public data from FDA openFDA, NLM RxNorm and CDC data services. It also uses a public historical pharmaceutical sales benchmark for older demand context.

If the machine has no internet connection, the application remains functional and explicitly reports sources as unavailable/cached.

## What is actually live

The public-data layer is real network ingestion when the runtime has internet access. It is not a fake timer or hardcoded JSON feed.

The following outputs now consume the live overlay:

- forecast
- projected inventory
- shortage timing
- reorder quantity
- inventory health
- risk probability
- risk level
- anomalies
- command-center ranking
- alerts
- scenario simulation baseline

## What is not claimed

The project does not claim to receive live private pharmacy/hospital transactions. A production customer must connect authorized facility sales, inventory, purchase-order and supplier feeds.

## Validation

```text
Python tests: 26/26 passed
UI smoke checks: 11/11 passed
```

Browser visual validation was not performed in this environment.

## Local medicine catalogue + cold-start behavior

The package now ships with **3,972 local medicine reference records** spanning 331 real active-ingredient names and reference formulation variants. The catalogue is separate from the six seeded facility medicines: catalogue size does not change inventory coverage or stock-out probability for an individual medicine.

On a network-enabled deployment, `python scripts/refresh_catalogue.py` can refresh the local catalogue from the current NLM RxNorm prescribable release. Selected medicines can also be enriched at runtime through the existing FDA/NLM live connectors.

### Inventory coverage fix

Coverage is now calculated from the actual stock quantity divided by positive daily demand, with forecast-horizon extrapolation when necessary. The calculation is independent of the number of catalogue records or monitored medicines. Therefore 56 units and 56,000 units cannot collapse to the same 0.5-day result unless their actual demand rates are also correspondingly different.
