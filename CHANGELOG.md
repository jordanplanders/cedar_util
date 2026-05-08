## 2026-05-08 - Codex

- `cedarkit/core/relationship.py`
  Refactored the relationship layer to support parallel calculation-facing and presentation-facing render channels. Added `pattern_calc` and `pattern_pres`, `*_calc` accessors, and calc/pres normalization mappings while preserving the directional meaning of `r1` and `r2`. Added directional evidence comments near the core pattern assignments.

- `tests/test_relationship_conventions.py`
  Added focused tests for the new calc/pres relationship behavior, including `r1` / `r2` preservation, calc-channel rendering, surrogate calc variants, and calc/pres mapping round-trips.

- `cedarkit/core/relationship.py`
  Flagged the duplicated `surr_r2xy` / `surr_r2yx` behavior as a later cleanup candidate rather than changing it in this parcel.

- `cedarkit/core/data_objects.py`
  Added a calc-facing relationship resolver and switched stored-output relation comparisons in the lag/metric selection path to use calc-facing relationship strings instead of presentation-facing ones. This change is limited to output-table filtering semantics and does not alter presentation-oriented summary/reporting accessors.

- Verification note
  This parcel was verified with compile checks and a focused resolver sanity check. Full end-to-end output filtering and visualization validation remain pending in later Cedarkit Part 1A parcels.

- Remaining Part 1A work
  `pull_df()` calc/pres transition handling and palette/parser alignment remain pending for later Cedarkit Part 1A parcels.
