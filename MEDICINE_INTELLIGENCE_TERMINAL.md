# StockShield AI — Medicine Intelligence Terminal

## Design objective

The Medicine Search surface is implemented as a search-time intelligence terminal rather than a simple autocomplete or catalogue lookup.

### Resolution pipeline

Query → normalization → local reference search → NLM RxNorm → FDA NDC → FDA labels → DailyMed → PubChem → optional web-index discovery → evidence merge → ranked identity → public supply/safety context → provenance packet.

## Evidence dimensions

- identity confidence
- source agreement
- evidence-record count
- formulation / route / manufacturer evidence where available
- FDA public shortage context
- FDA enforcement/recall context
- FDA label presence
- RxNorm identity evidence
- PubChem chemical identity evidence
- historical benchmark context where a category mapping exists
- local StockShield reference matches
- source freshness / cached status

## Quant-style design principles

1. Every derived indicator has a defined input boundary.
2. Public-web evidence is not silently converted into facility inventory.
3. Missing data remains missing.
4. Source agreement increases confidence only within a bounded range.
5. Search results preserve provenance.
6. Public signal pressure is a contextual indicator, not a clinical probability.
7. Facility stock-out probability remains dependent on authorized facility telemetry.
8. Cached data is explicitly marked cached.
9. Live registry failure does not destroy useful local reference evidence.
10. No arbitrary web page is treated as authoritative merely because it ranks highly.

## 3D intelligence field

The browser renders a dependency-free Canvas 3D particle/sphere field representing identity, evidence, supply and safety dimensions. It is intentionally visual analytics rather than decorative 3D: node density and brightness are driven by returned evidence indicators, while pointer rotation and wheel zoom allow inspection.

## No-fabrication boundary

A medicine identity can be resolved from public registries without knowing a pharmacy's current stock. The system therefore separates:

- **Reference / public intelligence:** globally searchable.
- **Facility analytics:** requires real inventory, sales, purchase orders, supplier and operational data.

A search result never invents facility stock, demand, reorder quantity, shortage date, or patient demand.
