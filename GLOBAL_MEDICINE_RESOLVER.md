# STOCKSHIELD AI — Global Medicine Retrieval & Resolution Engine

## Purpose

StockShield now includes an embedded **Global Medicine Retrieval & Resolution Engine (GM-RRE)**. It is a search-time federated retrieval/ranking layer, separate from the demand-forecasting model.

A user can enter an arbitrary medicine, active ingredient, brand, strength, or product name. The engine queries multiple public medicine registries in parallel, merges evidence, ranks candidate identities, and returns source provenance.

It does **not** pretend to crawl hundreds of thousands of websites on every query. That would be slow, fragile, and inaccurate. Instead it uses authoritative registries as the primary evidence layer and an optional web-search index as a last-mile discovery layer.

## Primary live sources

1. **NLM RxNorm** — medicine concepts, identifiers, current/historical relationships and approximate-name matching.
2. **FDA NDC Directory** — marketed U.S. drug listing/product identifiers and harmonized fields.
3. **FDA drug labels** — structured label evidence.
4. **DailyMed** — current drug-label/product evidence.
5. **NCBI PubChem** — chemical identity enrichment.
6. **Optional Brave Search API** — broad web-index discovery when `STOCKSHIELD_BRAVE_API_KEY` is configured.
7. **DuckDuckGo HTML search fallback** — broad discovery when no Brave key is configured; search results are explicitly labelled as discovery evidence, not authoritative approval evidence.

NLM states that the RxNorm API covers current and historical concepts, and the current prescribable RxNorm release can be downloaded automatically. FDA states that the NDC Directory is updated daily; FDA also warns that NDC listing information is submitted by labelers and is not itself a determination of FDA approval.

## Search flow

```text
User query
   ↓
Local 3,972-record reference index
   ↓
GM-RRE query normalizer
   ↓
Parallel live retrieval
   ├── NLM RxNorm approximate match
   ├── NLM RxNorm product search
   ├── FDA NDC
   ├── FDA labels
   ├── DailyMed
   └── PubChem
   ↓
Identity normalization + evidence merge
   ↓
Agreement / exactness scoring
   ↓
Ranked medicine candidates + provenance
   ↓
Optional web-index discovery if authoritative registries return no candidate
```

## Accuracy design

The engine does not return a single name simply because one web page mentions it. A candidate becomes stronger when:

- the normalized name matches the query exactly;
- multiple independent public sources identify the same medicine/product;
- an RxCUI, NDC, DailyMed set ID or other identifier agrees;
- the result is supported by multiple evidence records.

The response exposes the evidence and source names so a reviewer can see why a result was ranked highly.

## API

### `GET /api/medicine-search?q=<medicine>`

Example:

```text
/api/medicine-search?q=rifaximin&limit=20&web=1
```

Returns:

- `results` — ranked normalized medicine candidates
- `match_score` — retrieval confidence score, not clinical certainty
- `sources` — registries supporting the candidate
- `evidence` — source identifiers and product fields
- `source_status` — live/cached/unavailable status for each connector
- `local_matches` — matches from the local catalogue
- `web_discovery` — search-index results when no authoritative match is returned

## Important boundary

A global medicine resolver can identify medicines and public product records. It cannot know a particular hospital/pharmacy's actual stock, sales, purchase orders, supplier lead time, substitutions, or patient demand unless that facility supplies the data.

Therefore:

**Public medicine identity → resolver**

**Facility inventory/demand → authorized facility data**

**Forecast/risk/reorder → StockShield intelligence engine**

This separation prevents public web data from being incorrectly treated as local inventory truth.
