## 2026-05-08 - Codex

- `cedarkit/core/relationship.py`
  Refactored the relationship layer to support parallel calculation-facing and presentation-facing render channels. Added `pattern_calc` and `pattern_pres`, `*_calc` accessors, and calc/pres normalization mappings while preserving the directional meaning of `r1` and `r2`. Added directional evidence comments near the core pattern assignments.

- `tests/test_relationship_conventions.py`
  Added focused tests for the new calc/pres relationship behavior, including `r1` / `r2` preservation, calc-channel rendering, surrogate calc variants, and calc/pres mapping round-trips.

- `cedarkit/core/relationship.py`
  Flagged the duplicated `surr_r2xy` / `surr_r2yx` behavior as a later cleanup candidate rather than changing it in this parcel.

- Pending follow-on work
  Relation-based filtering, `pull_df()` calc/pres transition handling, and palette/parser alignment are intentionally not included in this changelog entry because they belong to later Cedarkit Part 1A parcels.
