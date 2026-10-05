# STOCKSHIELD AI — Live Data Layer

## Purpose

StockShield separates **facility truth** from **public live intelligence**.

Public sources can add current supply, safety, identity, and epidemiology context. They do not represent a pharmacy's private inventory or patient transactions.

## Current public connectors

| Source | Role | Refresh | Model use |
|---|---|---:|---|
| FDA Drug Shortages / openFDA | National supply-pressure signal | Daily | Bounded operational-risk adjustment |
| FDA Drug Enforcement / recalls | Safety/supply disruption context | Weekly | Bounded operational-risk adjustment |
| FDA NDC Directory | Current product/packaging identity | Daily | Identity/catalogue verification |
| NLM RxNorm | Standardized medicine identity | Current API | Identity normalization |
| CDC respiratory surveillance | Current epidemiology context | Source-dependent | Bounded demand-context adjustment for relevant categories |
| Public Pharma Sales benchmark | Long historical demand context | Download/cache | Bounded demand-context adjustment |

## Data propagation

```text
Base temporal demand model
        +
Historical public benchmark trend
        +
Relevant current epidemiology signal
        |
        v
Bounded demand-context overlay
        |
        v
Forecast -> inventory projection -> shortage timing -> reorder

FDA shortage / recall signals
        |
        v
Bounded operational-risk overlay
        |
        v
Risk -> priority -> alerts -> command center
```

The overlay is intentionally capped. It is an evidence layer, not a claim of causal or clinical certainty.

## Source behavior

- Fresh responses are cached locally.
- Cache TTL is configurable with `STOCKSHIELD_LIVE_TTL` (default 900 seconds).
- A network circuit breaker prevents repeated blocking when the environment has no internet access.
- Failed sources are shown as unavailable/cached; values are never fabricated.
- `POST`/write operations are not performed against external public sources.

## API

```text
GET /api/live-status
GET /api/live-refresh
GET /api/live-intelligence/{medicine_id}
GET /api/forecast/{medicine_id}
GET /api/risk/{medicine_id}
GET /api/recommendations/{medicine_id}
GET /api/inventory-projection/{medicine_id}
```

## Accuracy boundary

The highest-quality local forecast requires authorized facility data:

- timestamped sales/dispensing transactions
- current inventory
- stock-out events
- purchase orders / inbound quantities
- supplier lead times
- substitutions
- expiry/batch information where available

Public FDA/CDC/NLM data cannot replace those facility records.

## Recommended production deployment

Run a scheduled ingestion worker outside the request path. Persist raw responses with source timestamps and hashes, normalize identifiers through RxNorm/NDC, then retrain/recalibrate on authorized facility demand history.
