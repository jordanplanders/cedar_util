import pandas as pd
import sys
import os
import numpy as np
import gc
import re
from pathlib import Path
import pyarrow as pa


pd.option_context('mode.use_inf_as_na', True)

try:
    from cedarkit.utils.routing import set_calc_path, set_output_path, check_location, sqlite_paths
    from cedarkit.utils.routing import check_csv
    from cedarkit.core.data_objects import *
    from cedarkit.viz.grids import GridCell
    from cedarkit.utils.io import *
    from cedarkit.core.project_config import load_config
    from cedarkit.utils.cli import get_parser
    from cedarkit.utils.tables import *
    from cedarkit.utils.cli import setup_logging, log_line

except ImportError:
    # Fallback: imports when running as a package
    from utils.routing.paths import set_calc_path, set_output_path, check_location, sqlite_paths
    from utils.routing.file_name_parsers import check_csv
    from core.data_objects import *
    from viz.grids import GridCell
    from utils.io.cloudjoblib import *
    from core.project_config import load_config
    from utils.cli.arg_parser import get_parser
    from utils.tables.parquet_tools import *
    from utils.cli.logging import setup_logging, log_line

import logging

logger = logging.getLogger(__name__)


_REL_SEPS_influence = [r"\s+causes\s+", r"\s+influences\s+"]
_REL_SEPS_operation = [r"\s*->\s*", r"\s*→\s*", r"\s*=>\s*", r"\s+reconstructs\s+"]


def _normalize_lag_list(value, default_max_lag=None):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or value == "":
        if default_max_lag in (None, "", 0):
            return []
        ml = int(abs(int(default_max_lag)))
        return list(range(-ml, ml + 1))
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return [int(v) for v in parts]

def _get_output_path(E_tau_pair, tmp_dir, attr_name, object_grid=None, output_obj=None):
    E, tau=E_tau_pair[0], E_tau_pair[1]
    attr_obj = None
    if (output_obj is None) & (object_grid is not None):
        try:
            output_obj = object_grid[(E, tau)].output
            attr_obj = getattr(output_obj, attr_name, None)
        except:
            pass

    if attr_obj is None:
        files = [file for file in os.listdir(tmp_dir) if f'E{E}_tau{tau}__{attr_name}' in file]  # for debugging: list files in tmp_dir to see if expected files are present
        print(files)
        if len(files)==1:
            return tmp_dir/files[0]
        elif len(files)>1:
            log_line(logger, f"Multiple files found for E={E}, tau={tau}, attr {attr_name}: {[str(f) for f in files]}. Unable to determine correct path.",
                     indent=0, log_type="warning")
            return None
    else:
        return getattr(attr_obj, "path", None)


def _path_exists(path_like):
    if path_like is None:
        return False
    try:
        return Path(path_like).exists()
    except Exception:
        return False


def _parse_relation_pair(rel):
    '''
    parses relationship and returns (lhs, rhs) with lhs as the forcing variable and rhs as the responding variable, if possible. Otherwise returns (None, None).
    '''
    if rel is None:
        return None, None
    rel = str(rel).strip()
    # causal langauge parsing ('causes', 'influences')
    for sep in _REL_SEPS_influence:
        parts = re.split(sep, rel, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            lhs, rhs = parts[0].strip(), parts[1].strip()
            if lhs and rhs:
                return lhs, rhs
    # operational parsing (reconstruct, arrow)
    for sep in _REL_SEPS_operation:
        parts = re.split(sep, rel, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            lhs, rhs = parts[1].strip(), parts[0].strip()
            if lhs and rhs:
                return lhs, rhs
    # m = re.match(r"^\s*(.*?)\s+(causes|influences)\s+(.*?)\s*$", rel, flags=re.IGNORECASE)
    # if m:
    #     return m.group(1).strip(), m.group(3).strip()
    return None, None


def _filter_table_to_config_vars(table, allowed_vars):
    def _row_count(obj):
        if obj is None:
            return 0
        if isinstance(obj, pa.Table):
            return obj.num_rows
        if isinstance(obj, pl.LazyFrame):
            return obj.collect().height
        if isinstance(obj, pl.DataFrame):
            return obj.height
        if isinstance(obj, pd.DataFrame):
            return len(obj)
        raise TypeError(f"Unsupported table type: {type(obj)}")

    if table is None or _row_count(table) == 0:
        return table

    allowed = {str(v).strip().lower() for v in allowed_vars if v is not None}
    if len(allowed) != 2:
        return table

    if isinstance(table, pa.Table):
        df = table.to_pandas()
        source_type = "arrow"
    elif isinstance(table, pl.LazyFrame):
        df = table.collect().to_pandas()
        source_type = "lazy"
    elif isinstance(table, pl.DataFrame):
        df = table.to_pandas()
        source_type = "polars"
    elif isinstance(table, pd.DataFrame):
        df = table.copy()
        source_type = "pandas"
    else:
        raise TypeError(f"Unsupported table type: {type(table)}")

    if len(df) == 0:
        return table

    if "forcing" in df.columns and "responding" in df.columns:
        lhs = df["forcing"].astype("string").str.strip().str.lower()
        rhs = df["responding"].astype("string").str.strip().str.lower()
    else:
        rel_col = "relation" if "relation" in df.columns else ("relation_0" if "relation_0" in df.columns else None)
        if rel_col is None:
            return table
        pairs = df[rel_col].apply(_parse_relation_pair)
        lhs = pairs.apply(lambda x: (x[0] if x is not None else None))
        rhs = pairs.apply(lambda x: (x[1] if x is not None else None))
        lhs = lhs.astype("string").str.strip().str.lower()
        rhs = rhs.astype("string").str.strip().str.lower()

    mask = lhs.isin(allowed) & rhs.isin(allowed) & (lhs != rhs)
    filtered_df = df[mask].copy()
    if source_type == "arrow":
        return pa.Table.from_pandas(filtered_df, preserve_index=False)
    if source_type == "lazy":
        return pl.from_pandas(filtered_df).lazy()
    if source_type == "polars":
        return pl.from_pandas(filtered_df)
    return filtered_df


def _apply_relationship_safeguard(output_obj, allowed_vars, name):
    def _row_count(obj):
        if obj is None:
            return 0
        if isinstance(obj, pa.Table):
            return obj.num_rows
        if isinstance(obj, pl.LazyFrame):
            return obj.collect().height
        if isinstance(obj, pl.DataFrame):
            return obj.height
        if isinstance(obj, pd.DataFrame):
            return len(obj)
        raise TypeError(f"Unsupported table type: {type(obj)}")

    if output_obj is None:
        return
    output_obj.get_table()
    before = _row_count(output_obj._full)
    output_obj._full = _filter_table_to_config_vars(output_obj._full, allowed_vars)
    after = _row_count(output_obj._full)
    if before != after:
        log_line(
            logger,
            f"Filtered {name} rows by config vars {sorted(list(allowed_vars))}: {before} -> {after}",
            indent=0,
            log_type="warning",
        )



def process_config(grp_info, E_i, tau_i, tmp_dir, output_location, config, existing_output=None, calc_delta_rho_table=True,
                   aggregate_libsize_table=True, calc_delta_rho_full=True, path_info=None, override_paths=False,
                   collector_path=None, discovery_fn=None, row_query_fn=None,
                   metric_lag_mode=None, smoothing_window=1, metric_relationship=None):
    '''
    Process a single (E, tau) configuration and return a GridCell object containing the results.
    Parameters:
        - grp_info: dict, containing 'E' and 'tau' keys for the configuration to process.
        - E_i: int, index of the embedding dimension. (this is liable to change in actual usage)
        - tau_i: int, index of the time delay. (this is liable to change in actual usage)
        - tmp_dir: Path, cache directory for intermediate files.
        - output_location: Path, directory where output files are stored.
        - config: configuration object containing settings for data processing.
        - existing_output: GroupOutput object, optional existing output to update.
        - calc_delta_rho_table: bool, whether to calculate delta rho statistics.
        - aggregate_libsize_table: bool, whether to aggregate library size statistics.
    Returns:
        - GridCell object containing the processed results for the given (E, tau) configuration.

    Uses:
        - DataGroup: to manage and retrieve data files for the given configuration.
        - OutputCollection: to aggregate and manage output data.
        - GridCell: to encapsulate the results for the grid cell corresponding to (E, tau).
    '''

    print(f'Processing E={grp_info["E"]}, tau={grp_info["tau"]}', output_location , file=sys.stdout, flush=True)

    test_grp = DataGroup(grp_info, tmp_dir=tmp_dir)
    print('\tgetting files', file=sys.stdout, flush=True)
    path_info = path_info or {}
    desired_paths = {
        "delta_rho_stats": path_info.get("delta_rho_stats"),
        "delta_rho_full": path_info.get("delta_rho_full"),
        "libsize_aggregated": path_info.get("libsize_aggregated"),
    }
    calc_requests = {
        "delta_rho_stats": bool(calc_delta_rho_table),
        "delta_rho_full": bool(calc_delta_rho_full),
        "libsize_aggregated": bool(aggregate_libsize_table),
    }

    grp_specs = test_grp.get_group_config()
    new_output_col = OutputCollection(in_table=[], grp_specs=grp_specs, tmp_dir=tmp_dir)

    for attr_name, requested_path in desired_paths.items():
        if requested_path is None:
            continue

        if not _path_exists(requested_path):
            log_line(logger, f"Provided path for {attr_name} does not exist: {requested_path}",
                     indent=0, log_type="warning")
            continue
        try:
            reused_output = Output(None, path=requested_path, tmp_dir=tmp_dir, outtype=attr_name)
            reused_output.get_table()
            setattr(new_output_col, attr_name, reused_output)
            if override_paths is False:
                calc_requests[attr_name] = False
            log_line(logger, f"Reused existing {attr_name} from {requested_path}",
                     indent=0, log_type="info")
        except Exception as e:
            log_line(logger, f"Unable to load provided path for {attr_name}: {requested_path}; {e}",
                     indent=0, log_type="warning")

    calc_delta_rho_table = calc_requests["delta_rho_stats"]
    calc_delta_rho_full = calc_requests["delta_rho_full"]
    aggregate_libsize_table = calc_requests["libsize_aggregated"]

    print('\tchecking if calculations are needed based on requested outputs and existing paths', file=sys.stdout, flush=True)
    print(f'\tcalc_requests: {calc_requests}', file=sys.stdout, flush=True)

    # NEW: only run calculations when at least one output has been explicitly requested
    if any(calc_requests.values()):
        output_collections = []

        if collector_path is not None:
            test_grp.get_files(config, collector_path,
                               discovery_fn=discovery_fn, row_query_fn=row_query_fn)
        else:
            test_grp.get_files(config, output_location,
                               file_name_pattern='E{E}_tau{tau}_lag{lag}', source='parquet')

        # print(f'\tfound {len(test_grp.file_list)} files for E={grp_info["E"]}, tau={grp_info["tau"]}', file=sys.stdout, flush=True)
        log_line(logger, f'\tfound {len(test_grp.file_list)} files for E={grp_info["E"]}, tau={grp_info["tau"]}',
                 indent=0,
                 log_type="debug")

        if len(test_grp.file_list) < 1:
            print("Skipping because no files found.")
            return

        for ij, groupconfig_file in enumerate(test_grp.file_list):
            name = ''
            try:
                name = groupconfig_file.output_path[0].name
            except Exception:
                name = groupconfig_file.output_path

            log_line(logger, f'\t1 processing file {ij + 1}/{len(test_grp.file_list)}: {name}', indent=0,
                     log_type="debug")
            output_col = groupconfig_file.pull_output(to_table=False)

            if calc_delta_rho_table or calc_delta_rho_full:
                log_line(logger,
                         f'\tcalculating delta rho for {name}; full_out {bool(calc_delta_rho_full)}, stats_out {bool(calc_delta_rho_table)}',
                         indent=0, log_type="debug")
                output_col = output_col.calc_delta_rho(
                    full_out=bool(calc_delta_rho_full),
                    stats_out=bool(calc_delta_rho_table),
                )

            if aggregate_libsize_table is True:
                output_col = output_col.aggregate_libsize()

            log_line(logger,
                     f'\tprocessed {name} with calc_delta_rho_table={calc_delta_rho_table}, '
                     f'calc_delta_rho_full={calc_delta_rho_full}, aggregate_libsize_table={aggregate_libsize_table}',
                     indent=0, log_type="debug")
            output_collections.append(output_col)

        computed_output_col = OutputCollection(in_table=output_collections, grp_specs=grp_specs, tmp_dir=tmp_dir)

        # NEW: recomputed outputs are authoritative whenever their flag is True
        if calc_requests["delta_rho_stats"]:
            new_output_col.delta_rho_stats = computed_output_col.delta_rho_stats
        if calc_requests["delta_rho_full"]:
            new_output_col.delta_rho_full = computed_output_col.delta_rho_full
        if calc_requests["libsize_aggregated"]:
            new_output_col.libsize_aggregated = computed_output_col.libsize_aggregated


    e_val = grp_info.get("E")
    tau_val = grp_info.get("tau")
    et_tag = f"E{e_val}_tau{tau_val}"
    df = pd.DataFrame(columns=["surr_var", "surr_num_count_distinct"])
    write_status = {
        "delta_rho_stats": "not_requested",
        "delta_rho_full": "not_requested",
        "libsize_aggregated": "not_requested",
    }

    if aggregate_libsize_table is False:
        libsize_aggregated_path = _get_output_path((e_val, tau_val),tmp_dir,  "libsize_aggregated", output_obj=existing_output)
        if libsize_aggregated_path is not None and et_tag in str(
                libsize_aggregated_path) and new_output_col.libsize_aggregated is not None:
            new_output_col.libsize_aggregated.path = libsize_aggregated_path

    if calc_delta_rho_table is False:
        delta_rho_path = _get_output_path((e_val, tau_val),tmp_dir,  "delta_rho_stats", output_obj=existing_output)
        if delta_rho_path is not None and et_tag in str(delta_rho_path) and new_output_col.delta_rho_stats is not None:
            new_output_col.delta_rho_stats.path = delta_rho_path

    if calc_delta_rho_full is False:
        delta_rho_path_full = _get_output_path((e_val, tau_val),tmp_dir,  "delta_rho_full", output_obj=existing_output)
        if delta_rho_path_full is not None and et_tag in str(
                delta_rho_path_full) and new_output_col.delta_rho_full is not None:
            new_output_col.delta_rho_full.path = delta_rho_path_full


    # Safeguard: keep only rows whose relationship is between the two vars declared in proj_config.
    allowed_vars = {config.col.var, config.target.var}
    _apply_relationship_safeguard(new_output_col.delta_rho_stats, allowed_vars, "delta_rho_stats")
    _apply_relationship_safeguard(new_output_col.delta_rho_full, allowed_vars, "delta_rho_full")
    _apply_relationship_safeguard(new_output_col.libsize_aggregated, allowed_vars, "libsize_aggregated")

    if metric_lag_mode is not None and new_output_col.delta_rho_stats is not None:
        try:
            new_output_col.calc_metrics(
                lag=metric_lag_mode,
                smoothing_window=smoothing_window,
                relationship_id=metric_relationship,
            )
            log_line(logger, f"calc_metrics completed for E={e_val}, tau={tau_val}", indent=0, log_type="info")
        except Exception as e:
            log_line(logger, f"calc_metrics failed for E={e_val}, tau={tau_val}: {e}", indent=0, log_type="warning")

    if aggregate_libsize_table is True and new_output_col.libsize_aggregated is not None:
        try:
            gb = new_output_col.libsize_aggregated.surrogate.group_by(["surr_var"]).aggregate([("surr_num", "count_distinct")])
            df = gb.to_pandas()
        except Exception as e:
            log_line(logger, f"Unable to build annotations from libsize_aggregated for E={e_val}, tau={tau_val}: {e}",
                     indent=0, log_type="warning")

    if calc_delta_rho_table is True:
        write_status["delta_rho_stats"] = "skipped_missing_object"
        if new_output_col.delta_rho_stats is not None:
            try:
                new_output_col.delta_rho_stats.write_table(tag=f"E{e_val}_tau{tau_val}__delta_rho_stats")
                write_status["delta_rho_stats"] = "written"
                log_line(logger, '\twriting delta rho stats table', indent=0, log_type="info")
            except Exception as e:
                write_status["delta_rho_stats"] = f"failed: {type(e).__name__}"
                log_line(logger, f"Failed writing delta rho stats for E={e_val}, tau={tau_val}: {e}",
                         indent=0, log_type="error")

    if calc_delta_rho_full is True:
        write_status["delta_rho_full"] = "skipped_missing_object"
        full_obj = new_output_col.delta_rho_full
        pre_rows = 0
        if full_obj is not None:
            try:
                full_obj.get_table()
                if isinstance(full_obj._full, pa.Table):
                    pre_rows = full_obj._full.num_rows
                elif isinstance(full_obj._full, pl.LazyFrame):
                    pre_rows = full_obj._full.collect().height
                elif isinstance(full_obj._full, pl.DataFrame):
                    pre_rows = full_obj._full.height
                elif isinstance(full_obj._full, pd.DataFrame):
                    pre_rows = len(full_obj._full)
                else:
                    pre_rows = 0 if full_obj._full is None else 0
                log_line(logger, f"delta_rho_full pre-write rows for E={e_val}, tau={tau_val}: {pre_rows}",
                         indent=0, log_type="info")
            except Exception as e:
                write_status["delta_rho_full"] = f"failed_precheck: {type(e).__name__}"
                log_line(logger, f"Failed pre-check for delta_rho_full E={e_val}, tau={tau_val}: {e}",
                         indent=0, log_type="error")

        if full_obj is not None and pre_rows > 0:
            try:
                full_obj.write_table(tag=f"E{e_val}_tau{tau_val}__delta_rho_full")
                out_path = full_obj.path
                if _path_exists(out_path):
                    write_status["delta_rho_full"] = "written"
                    log_line(logger, f"\twriting delta rho full table -> {out_path}", indent=0, log_type="info")
                else:
                    write_status["delta_rho_full"] = "failed_file_missing_after_write"
                    log_line(logger, f"delta_rho_full path missing after write for E={e_val}, tau={tau_val}: {out_path}",
                             indent=0, log_type="error")
            except Exception as e:
                write_status["delta_rho_full"] = f"failed: {type(e).__name__}"
                log_line(logger, f"Failed writing delta rho full for E={e_val}, tau={tau_val}: {e}",
                         indent=0, log_type="error")
        elif full_obj is not None and pre_rows == 0 and "failed_precheck" not in write_status["delta_rho_full"]:
            write_status["delta_rho_full"] = "skipped_empty_table"
            log_line(logger, f"Skipping delta_rho_full write for E={e_val}, tau={tau_val}: empty table",
                     indent=0, log_type="warning")

    if aggregate_libsize_table is True:
        write_status["libsize_aggregated"] = "skipped_missing_object"
        if new_output_col.libsize_aggregated is not None:
            try:
                new_output_col.libsize_aggregated.write_table(tag=f"E{e_val}_tau{tau_val}__libsize_aggregated")
                write_status["libsize_aggregated"] = "written"
                log_line(logger, '\twriting libsize aggregated table', indent=0, log_type="info")
            except Exception as e:
                write_status["libsize_aggregated"] = f"failed: {type(e).__name__}"
                log_line(logger, f"Failed writing libsize_aggregated for E={e_val}, tau={tau_val}: {e}",
                         indent=0, log_type="error")

    # print('\tclearing tables', file=sys.stdout, flush=True)
    log_line(logger, '\tclearing tables',
             indent=0,
             log_type="info")
    new_output_col.clear_tables()

    cell_obj = GridCell(E_i, tau_i, new_output_col)
    cell_obj.write_status = write_status
    del new_output_col

    cell_obj.row_labels.append(f"E={grp_info['E']}")
    cell_obj.col_labels.append(f"tau={grp_info['tau']}")

    for _, row in df.iterrows():
        cell_obj.annotations.append(f"{row['surr_var']}: n={row['surr_num_count_distinct']}")

    cell_obj.occupied = True
    return cell_obj


if __name__ == "__main__":
    ''' 
    Command line interface for processing (E, tau) configurations and generating object grid files.
    Uses argparse to parse command line arguments for project name, file names, temporary directory, indices, and flags.
    1. Parses command line arguments for project name, object grid file name, group file name, temporary directory, indices, and flags.
    2. Loads project configuration from YAML file.
    3. Reads e_tau_grps_df from specified CSV file.
    4. For each specified (E, tau) configuration, processes the configuration using process_config function.
    5. Saves the resulting object grid to a joblib file in the temporary directory.
    6. Skips processing for configurations that have already been processed unless specific flags are set.
    7. Outputs progress and status messages to stdout.
    
    '''
    setup_logging()

    parser = get_parser()
    args = parser.parse_args()



    if args.project is not None:
        proj_name = args.project
    else:
        log_line(logger, 'project name is required',
                 indent=0,
                 log_type="info")
        log_line(logger, 'project name is required',
                 indent=0,
                 log_type="error")
        # print('project name is required', file=sys.stdout, flush=True)
        # print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # When run from the command line, assumes that the current working directory is the directory containing the proj_name (dyad) directory e.g. hol_temp_tsi_ccm
    if args.proj_dir is not None:
        proj_dir = Path(args.proj_dir) / proj_name
    else:
        proj_dir = Path(os.getcwd()) / proj_name

    gen_config = 'proj_config'
    config = load_config(proj_dir / f'{gen_config}.yaml')

    obj_grid_file_name = args.file if args.file is not None else f'{proj_name}_obj_grid.joblib'
    group_file_name = args.group_file if args.group_file is not None else config.csvs.e_tau_grps
    tmp_dir = args.dir if args.dir is not None else'tmp' #target directory for cell object files and object_grid

    if args.inds is not None:
        ind = int(args.inds[-1])
    else:
        ind = int(sys.argv[-1])

    calc_delta_rho_table = False
    aggregate_libsize_table = False
    calc_delta_rho_table_full = False
    if args.flags is not None:
        if 'calc_delta_rho' in args.flags:
            calc_delta_rho_table = True
        if 'aggregate_libsize' in args.flags:
            aggregate_libsize_table = True
        if 'calc_delta_rho_full' in args.flags:
            calc_delta_rho_table_full = True
        print(args.flags, file=sys.stdout, flush=True)

    calc_location = set_calc_path(args, proj_dir, config)
    log_line(logger, f'Calculation location: {calc_location}', indent=0, log_type="info")
    # print(f'Calculation location: {calc_location}', file=sys.stdout, flush=True)
    # print(f'Read e_tau_grps_df from {group_file_name}.', file=sys.stdout, flush=True)
    log_line(logger, f'Read e_tau_grps_df from {group_file_name}.', indent=0, log_type="info")

    e_tau_grps_df = pd.read_csv(calc_location / check_csv(group_file_name))

    source = getattr(args, 'source', 'parquet') or 'parquet'
    if source == 'parquet':
        output_location = resolve_consolidated_dir(calc_location, config, 'parquet')
        sqlite_dir = None
    else:
        output_location = None
        sqlite_dir = resolve_consolidated_dir(calc_location, config, 'sqlite')

    tmp_dir = proj_dir / tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # this is hardcoded but should be released and left to the construction of the e_tau_grps_df
    E_vals = [4, 5, 6, 7, 8, 9, 10]
    tau_vals = [1, 2, 3, 4, 5, 6, 7, 8]
    comb_df = e_tau_grps_df[e_tau_grps_df['E'].isin(E_vals) & e_tau_grps_df['tau'].isin(tau_vals)].copy()
    comb_plot_df = comb_df.drop_duplicates()
    comb_plot_df = comb_plot_df.sort_values(by=['col_var_id', 'target_var_id', 'E', 'tau'])

    row = comb_plot_df.iloc[ind].to_dict()
    E = row['E']
    tau = row['tau']

    # Build sqlite callables after row is available (collector path depends on dyad name)
    src_kwargs = {}
    if source == 'sqlite':
        output_dir = set_output_path(args, calc_location, config)
        _sqlite_dir, collector_path, _run_db_path = sqlite_paths(
            proj_dir,
            config,
            calc_location=calc_location,
            output_dir=output_dir,
            ensure=False,
        )
        log_line(logger, f'sqlite collector: {collector_path}', indent=0, log_type="info")

        col_var_name = str(row.get("col_var") or getattr(getattr(config, "col", None), "var", "") or "")
        target_var_name = str(row.get("target_var") or getattr(getattr(config, "target", None), "var", "") or "")
        lag_list = _normalize_lag_list(row.get("lags"), default_max_lag=row.get("max_lag"))

        def _discovery_fn(cp, grp_d):
            return [grp_d]

        def _row_query_fn(td):
            cause_name_expr = "COALESCE(NULLIF(vca.name, ''), NULLIF(vci.cedarkit_var_id, ''), '')"
            effect_name_expr = "COALESCE(NULLIF(veb.name, ''), NULLIF(vei.cedarkit_var_id, ''), '')"

            clauses = []
            params = {
                "E": int(td["E"]),
                "tau": int(td["tau"]),
                "knn": int(td.get("knn") or 0),
                "col_var_name": col_var_name,
                "target_var_name": target_var_name,
            }

            clauses.append("mc_cause.E = :E")
            clauses.append("mc_cause.tau = :tau")
            clauses.append("rss.metric = 'corr'")
            clauses.append(f"{effect_name_expr} IN (:col_var_name, :target_var_name)")
            clauses.append(f"{cause_name_expr} IN (:col_var_name, :target_var_name)")
            clauses.append(f"{cause_name_expr} != {effect_name_expr}")

            tp_value = td.get("Tp", td.get("tp"))
            if tp_value not in (None, ""):
                clauses.append("rss.tp = :Tp")
                params["Tp"] = int(tp_value)

            draw_size = td.get("draw_size", td.get("sample"))
            if draw_size not in (None, "", 0):
                clauses.append("s.draw_size = :draw_size")
                params["draw_size"] = int(draw_size)

            max_lag = td.get("max_lag")
            if max_lag not in (None, "", 0):
                clauses.append("s.max_lag = :max_lag")
                params["max_lag"] = int(max_lag)

            if lag_list:
                lag_placeholders = []
                for i, lag in enumerate(lag_list):
                    key = f"lag_{i}"
                    lag_placeholders.append(f":{key}")
                    params[key] = int(lag)
                clauses.append(f"rss.lag IN ({', '.join(lag_placeholders)})")

            # there is a convention that r1 is x reconstructs y, so x is by default effect and y is cause.
            # this appears not be problematic when both directions are being queried together.
            sql = f"""
            SELECT
              mc_cause.E AS E,
              mc_cause.tau AS tau,
              rss.tp AS Tp,
              rss.lag AS lag,
              :knn AS knn,
              CASE
                WHEN COALESCE(vci.surrogate_number, 0) = 0 AND COALESCE(vei.surrogate_number, 0) = 0 THEN 'neither'
                WHEN COALESCE(vci.surrogate_number, 0) > 0 AND COALESCE(vei.surrogate_number, 0) = 0 THEN {cause_name_expr}
                WHEN COALESCE(vci.surrogate_number, 0) = 0 AND COALESCE(vei.surrogate_number, 0) > 0 THEN {effect_name_expr}
                ELSE 'both'
              END AS surr_var,
              MAX(COALESCE(vci.surrogate_number, 0), COALESCE(vei.surrogate_number, 0)) AS surr_num,
              {effect_name_expr} AS x_id,
              0 AS x_age_model_ind,
              {effect_name_expr} AS x_var,
              {cause_name_expr} AS y_id,
              0 AS y_age_model_ind,
              {cause_name_expr} AS y_var,
              s.draw_size AS LibSize,
              ROW_NUMBER() OVER (
                ORDER BY rss.run_id, rss.draw_id, rss.lag, {cause_name_expr}, {effect_name_expr}
              ) - 1 AS ind_i,
              {effect_name_expr} || ' -> ' || {cause_name_expr} AS relation,
              {cause_name_expr} AS forcing,
              {effect_name_expr} AS responding,
              rss.run_id AS run_id,
              rss.draw_id AS draw_id,
              rss.value AS rho
            FROM reconstruction_sampling_stat rss
            JOIN sampling s ON s.draw_id = rss.draw_id
            JOIN manifold_config mc_cause ON mc_cause.id = rss.cause_manifold_id
            JOIN manifold_config mc_effect ON mc_effect.id = rss.effect_manifold_id
            LEFT JOIN compound_variable_member cvmc ON cvmc.compound_variable_id = mc_cause.compound_variable_id
            LEFT JOIN compound_variable_member cvme ON cvme.compound_variable_id = mc_effect.compound_variable_id
            LEFT JOIN variable_instance vci ON vci.id = cvmc.variable_instance_id
            LEFT JOIN variable_instance vei ON vei.id = cvme.variable_instance_id
            LEFT JOIN variable vca ON vca.id = vci.variable_id
            LEFT JOIN variable veb ON veb.id = vei.variable_id
            """
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += f" ORDER BY rss.run_id, rss.draw_id, rss.lag, {cause_name_expr}, {effect_name_expr}"
            return sql, params

        src_kwargs = dict(collector_path=collector_path,
                          discovery_fn=_discovery_fn,
                          row_query_fn=_row_query_fn)

    metric_lag_mode = getattr(args, 'metric_lag_mode', None)
    smoothing_window = getattr(args, 'smoothing_window', 1) or 1
    metric_relationship = getattr(args, 'metric_relationship', None)
    if metric_lag_mode is not None:
        try:
            metric_lag_mode = int(metric_lag_mode)
        except (ValueError, TypeError):
            pass  # keep as string ('pos' / 'neg')
    src_kwargs.update(
        metric_lag_mode=metric_lag_mode,
        smoothing_window=smoothing_window,
        metric_relationship=metric_relationship,
    )

    E_is = {E: ik for ik, E in enumerate(np.arange(min(E_vals), max(E_vals) + 1))}
    tau_is = {tau: ik for ik, tau in enumerate(np.arange(min(tau_vals), max(tau_vals) + 1))}


    try:
        object_grid = joblib_cloud_load(tmp_dir / obj_grid_file_name)
    except:
        object_grid = {}

    # Process the (E, tau) configuration if not already processed
    not_in_grid = (E, tau) not in object_grid.keys()
    output_is_none = (not_in_grid is False) and ((object_grid[(E, tau)] is None) or (object_grid[(E, tau)].output is None))

    # print(f'E{E}-tau{tau}; not_in_grid: {not_in_grid}, output_is_none: {output_is_none}', file=sys.stdout, flush=True)
    stats_path = _get_output_path((E, tau),  tmp_dir, "delta_rho_stats", object_grid=object_grid)
    full_path = _get_output_path( (E, tau),  tmp_dir, "delta_rho_full", object_grid=object_grid)
    libsize_path = _get_output_path( (E, tau),  tmp_dir, "libsize_aggregated", object_grid=object_grid)
    path_info = {attr: _get_output_path( (E, tau),  tmp_dir, attr, object_grid=object_grid) for attr in ["delta_rho_stats", "delta_rho_full", "libsize_aggregated"]}
    path_info = {key:value for key, value in path_info.items() if value is not None}
    print(path_info, file=sys.stdout, flush=True)

    if not_in_grid is True or output_is_none is True:
        # print('regardless of flags, going the dual calculations', file=sys.stdout, flush=True)
        log_line(logger, 'regardless of flags, going the dual calculations', indent=0,
                 log_type="info")
        if args.flags is None:
            calc_delta_rho_table = True
            aggregate_libsize_table = True
            calc_delta_rho_table_full = True
        object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config, calc_delta_rho_table=calc_delta_rho_table,
                                               aggregate_libsize_table=aggregate_libsize_table, calc_delta_rho_full=calc_delta_rho_table_full, path_info=path_info,
                                               **src_kwargs)

        joblib_cloud_atomic_dump(object_grid, tmp_dir / obj_grid_file_name, compress=3,
                                 protocol=5)
        gc.collect()
        log_line(logger, f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", indent=0,
                 log_type="info")
        # print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
    else:
        if object_grid[(E, tau)].output is None:
            if args.flags is None:
                calc_delta_rho_table = True
                aggregate_libsize_table = True
                calc_delta_rho_table_full = True
            log_line(logger, 'output is None, going the dual calculations', indent=0,
                     log_type="info")
            # print('output is None, going the dual calculations', file=sys.stdout, flush=True)
        else:
            # stats_path = _get_output_path(object_grid[(E, tau)].output, "delta_rho_stats")
            # full_path = _get_output_path(object_grid[(E, tau)].output, "delta_rho_full")
            # libsize_path = _get_output_path(object_grid[(E, tau)].output, "libsize_aggregated")
            if (object_grid[(E, tau)].output.delta_rho_stats is None) or (stats_path is None) or (not _path_exists(stats_path)):
                calc_delta_rho_table = True
            if (object_grid[(E, tau)].output.delta_rho_full is None) or (full_path is None) or (not _path_exists(full_path)):
                calc_delta_rho_table_full = True
            if (object_grid[(E, tau)].output.libsize_aggregated is None) or (libsize_path is None) or (not _path_exists(libsize_path)):
                aggregate_libsize_table = True

        log_line(logger, ['calculations have been explicitly set: calc_delta_rho_table', calc_delta_rho_table,
              '; aggregate_libsize:', aggregate_libsize_table], indent=0,
                 log_type="info")
        # print('calculations have been explicitly set: calc_delta_rho_table', calc_delta_rho_table,
        #       '; aggregate_libsize:', aggregate_libsize_table, file=sys.stdout, flush=True)
        if bool(path_info) or (calc_delta_rho_table is True) or (aggregate_libsize_table is True) or (calc_delta_rho_table_full is True):

            object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config,
                                                   existing_output=object_grid[(E, tau)].output,
                                                       calc_delta_rho_table=calc_delta_rho_table,
                                                       aggregate_libsize_table=aggregate_libsize_table,
                                                   calc_delta_rho_full=calc_delta_rho_table_full, path_info=path_info,
                                                   **src_kwargs)

            joblib_cloud_atomic_dump(object_grid, tmp_dir/obj_grid_file_name, compress=3,
                                   protocol=5)
            gc.collect()
            log_line(logger, f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", indent=0,
                     log_type="info")
            # print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
        else:
            log_line(logger, f"Skipping E={E}, tau={tau} because already processed.", indent=0,
                     log_type="info")
            # print(f"Skipping E={E}, tau={tau} because already processed.", file=sys.stdout, flush=True)

    # Per-cell summary for bash-run diagnostics
    try:
        if (E, tau) in object_grid and object_grid[(E, tau)] is not None and object_grid[(E, tau)].output is not None:
            out = object_grid[(E, tau)].output
            full_path = _get_output_path((E,tau), tmp_dir, "delta_rho_full", object_grid=object_grid)
            full_exists = _path_exists(full_path)
            full_requested = bool(calc_delta_rho_table_full) or bool(not_in_grid) or bool(output_is_none)
            full_status = "written" if full_exists else ("failed" if full_requested else "skipped")
            extra = ""
            if hasattr(object_grid[(E, tau)], "write_status"):
                extra = f"; write_status={object_grid[(E, tau)].write_status.get('delta_rho_full')}"
            log_line(
                logger,
                f"Cell summary E={E}, tau={tau}: full_requested={full_requested}; full_status={full_status}; full_path={full_path}{extra}",
                indent=0,
                log_type="info",
            )
        else:
            log_line(
                logger,
                f"Cell summary E={E}, tau={tau}: full_requested=True; full_status=failed; reason=no_output_cell",
                indent=0,
                log_type="warning",
            )
    except Exception as e:
        log_line(logger, f"Failed to emit cell summary for E={E}, tau={tau}: {e}", indent=0, log_type="error")
