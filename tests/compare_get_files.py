"""
Shadow comparison: DataGroup.get_files (Polars) vs _get_files_v1 (PyArrow).
Run with:
    ~/miniconda3/envs/hol_ccm_local_cedar_env/bin/python tests/compare_get_files.py

Edit GRP_D and PATHS at the top to match your project.
"""
import sys
from pathlib import Path

# ── configure these ──────────────────────────────────────────────────────────
PROJ_DIR = Path("/Users/jlanders/PycharmProjects/hol_temp_tsi_ccm/GISP2Alley00Tanom_Wu18TSI")
OUTPUT_PATH = PROJ_DIR / "calc_local_tmp" / "calc_refactor" / "parquet"
PROJ_CONFIG_PATH = PROJ_DIR / "proj_config.yaml"

# narrow grp_d (specific surr_num) — tests targeted queries
GRP_D_NARROW = {
    "E": 4,
    "tau": 1,
    "knn": 20,
    "Tp": 1,
    "surr_var": "neither",
    "surr_num": 0,
    'col_var_id':'GISP2Alley00Tanom',
    'target_var_id':'Wu18TSI'
}

# broad grp_d (all lags, all surr for one E/tau) — tests memory-bound path
GRP_D_BROAD = {
    "E": 4,
    "tau": 1,
    "knn": 20,
    "Tp": 1,
    'col_var_id': 'GISP2Alley00Tanom',
    'target_var_id': 'Wu18TSI'
}
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))

from cedarkit.core.data_objects import DataGroup


def local_load_config(proj_config_path):
    try:
        from cedarkit.core.project_config import ProjectConfig, load_config
        return load_config(proj_config_path)
    except Exception as e:
        print(f"Could not load config: {e}")
        sys.exit(1)


def sorted_paths(file_list):
    return sorted(str(rc.output_path[0]) for rc in file_list)


def compare(label, grp_d, config):
    print(f"\n{'='*60}")
    print(f"Comparing: {label}")
    print(f"grp_d: {grp_d}")
    print('='*60)

    dg_old = DataGroup(grp_d)
    dg_old._get_files_v1(config, OUTPUT_PATH)

    dg_new = DataGroup(grp_d)
    dg_new.get_files(config, OUTPUT_PATH)

    old_paths = sorted_paths(dg_old.file_list)
    new_paths = sorted_paths(dg_new.file_list)

    print(f"  _get_files_v1 found: {len(old_paths)} files")
    print(f"  get_files     found: {len(new_paths)} files")

    path_diff = set(old_paths) ^ set(new_paths)
    # if path_diff:
    #     print(f"  MISMATCH — symmetric diff: {path_diff}")
    # else:
    #     print("  File paths match ✓")

    # per-file trait dict comparison
    trait_mismatches = 0
    for path in old_paths:
        old_rc = next(rc for rc in dg_old.file_list if str(rc.output_path[0]) == path)
        new_rc = next((rc for rc in dg_new.file_list if str(rc.output_path[0]) == path), None)
        if new_rc is None:
            # print(f"  MISSING in new: {path}")
            trait_mismatches += 1
            continue
        old_d = {k: sorted(v) if isinstance(v, list) else v for k, v in old_rc.to_dict().items()}
        new_d = {k: sorted(v) if isinstance(v, list) else v for k, v in new_rc.to_dict().items()}
        if old_d != new_d:
            diff_keys = {k for k in set(old_d) | set(new_d) if old_d.get(k) != new_d.get(k)}
            print(f"  TRAIT DIFF for {Path(path).name}:")
            for k in diff_keys:
                print(f"    {k}: old={old_d.get(k)}  new={new_d.get(k)}")
            trait_mismatches += 1

    if trait_mismatches == 0:
        print("  Trait dicts match ✓")

    # static/nonstatic promotion
    if dg_old.static_traits == dg_new.static_traits:
        print("  static_traits match ✓")
    else:
        diff = {k for k in set(dg_old.static_traits) | set(dg_new.static_traits)
                if dg_old.static_traits.get(k) != dg_new.static_traits.get(k)}
        print(f"  static_traits MISMATCH on keys: {diff}")

    # pull_output row count (only if file lists match)
    if not path_diff:
        print("  Comparing pull_output row counts...")
        oc_old = dg_old._pull_output_v1()
        oc_new = dg_new.pull_output()
        old_rows = oc_old.table.full.num_rows if oc_old.table is not None else 0
        new_rows = oc_new.table.full.num_rows if oc_new.table is not None else 0
        print(f"    _pull_output_v1: {old_rows} rows")
        print(f"    pull_output:     {new_rows} rows")
        if old_rows == new_rows:
            print("  Row counts match ✓")
        else:
            print(f"  ROW COUNT MISMATCH: {old_rows} vs {new_rows}")


if __name__ == "__main__":
    config = local_load_config(PROJ_CONFIG_PATH)
    compare("narrow (specific surr_num=0)", GRP_D_NARROW, config)
    # compare("broad (all lags/surr for E=4 tau=1)", GRP_D_BROAD, config)
    print("\nDone.")
