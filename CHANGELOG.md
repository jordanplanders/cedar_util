## 2026-05-08 - Codex

- `cedarkit/core/relationship.py`
  Refactored the relationship layer to support parallel calculation-facing and presentation-facing render channels. Added `pattern_calc` and `pattern_pres`, `*_calc` accessors, and calc/pres normalization mappings while preserving the directional meaning of `r1` and `r2`. Added directional evidence comments near the core pattern assignments.

- `tests/test_relationship_conventions.py`
  Added focused tests for the new calc/pres relationship behavior, including `r1` / `r2` preservation, calc-channel rendering, surrogate calc variants, and calc/pres mapping round-trips.

- `cedarkit/core/relationship.py`
  Flagged the duplicated `surr_r2xy` / `surr_r2yx` behavior as a later cleanup candidate rather than changing it in this parcel.

- `cedarkit/core/data_objects.py`
  Added a calc-facing relationship resolver and switched stored-output relation comparisons in the lag/metric selection path to use calc-facing relationship strings instead of presentation-facing ones. This change is limited to output-table filtering semantics and does not alter presentation-oriented summary/reporting accessors.

- `cedarkit/viz/panels.py`
  Updated `pull_df()` so relation-family filtering can accept both calc-facing and presentation-facing spellings for `r1` and `r2`, then normalize the returned dataframe relation column to the requested convention. The default visualization-facing return convention remains presentation-oriented, with an explicit calc option available for callers that need it.

- Verification note
  The relationship and data-object parcels were verified with compile checks and focused resolver sanity checks. The `panels.py` parcel compiles cleanly, but runtime `pull_df()` verification is still pending here because this environment does not currently have `pyarrow` available.

- Remaining Part 1A work
  Palette/parser alignment remains pending for a later Cedarkit Part 1A parcel.
