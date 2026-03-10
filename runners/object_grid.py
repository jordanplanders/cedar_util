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
    from cedarkit.utils.routing import set_calc_path, set_output_path, check_location
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
    from utils.routing.paths import set_calc_path, set_output_path, check_location
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


_REL_SEPS = [r"\s*->\s*", r"\s*→\s*", r"\s*=>\s*", r"\s+causes\s+", r"\s+influences\s+"]


def _parse_relation_pair(rel):
    if rel is None:
        return None, None
    rel = str(rel).strip()
    for sep in _REL_SEPS:
        parts = re.split(sep, rel, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            lhs, rhs = parts[0].strip(), parts[1].strip()
            if lhs and rhs:
                return lhs, rhs
    m = re.match(r"^\s*(.*?)\s+(causes|influences)\s+(.*?)\s*$", rel, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(3).strip()
    return None, None


def _filter_table_to_config_vars(table, allowed_vars):
    if table is None or table.num_rows == 0:
        return table

    allowed = {str(v).strip().lower() for v in allowed_vars if v is not None}
    if len(allowed) != 2:
        return table

    df = table.to_pandas()
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
    return pa.Table.from_pandas(filtered_df, preserve_index=False)


def _apply_relationship_safeguard(output_obj, allowed_vars, name):
    if output_obj is None:
        return
    output_obj.get_table()
    before = 0 if output_obj._full is None else output_obj._full.num_rows
    output_obj._full = _filter_table_to_config_vars(output_obj._full, allowed_vars)
    after = 0 if output_obj._full is None else output_obj._full.num_rows
    if before != after:
        log_line(
            logger,
            f"Filtered {name} rows by config vars {sorted(list(allowed_vars))}: {before} -> {after}",
            indent=0,
            log_type="warning",
        )


def process_config(grp_info, E_i, tau_i, tmp_dir, output_location, config, existing_output=None, calc_delta_rho_table=True,
                   aggregate_libsize_table=True, calc_delta_rho_full=True):
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

    test_grp.get_files(config, output_location ,
                       file_name_pattern='E{E}_tau{tau}_lag{lag}', source='parquet')

    # print(f'\tfound {len(test_grp.file_list)} files for E={grp_info["E"]}, tau={grp_info["tau"]}', file=sys.stdout, flush=True)
    log_line(logger, f'\tfound {len(test_grp.file_list)} files for E={grp_info["E"]}, tau={grp_info["tau"]}', indent=0,
             log_type="debug")
    if len(test_grp.file_list) < 1:
        print("Skipping because no files found.")
        return

    output_collections = []
    for ij, groupconfig_file in enumerate(test_grp.file_list):
        name = ''
        try:
            name = groupconfig_file.output_path[0].name
        except:
            name = groupconfig_file.output_path

        # print(f'\t1 processing file {ij + 1}/{len(test_grp.file_list)}: {name}', file=sys.stdout, flush=True)
        log_line(logger, f'\t1 processing file {ij + 1}/{len(test_grp.file_list)}: {name}', indent=0,
                 log_type="debug")
        output_col = groupconfig_file.pull_output(to_table=False)

        full_out = False
        stats_out = True
        if calc_delta_rho_full is True:
            full_out = True

        if calc_delta_rho_table is True:
            stats_out = True
        if (full_out is True) or (stats_out is True):
            # print(f'\tcalculating delta rho for {name}; full_out {full_out}, stats_out {stats_out}', file=sys.stdout, flush=True)
            log_line(logger, f'\tcalculating delta rho for {name}; full_out {full_out}, stats_out {stats_out}', indent=0,
                     log_type="debug")
            output_col = output_col.calc_delta_rho(full_out=full_out, stats_out=stats_out)

        if aggregate_libsize_table is True:
            output_col = output_col.aggregate_libsize()

        # print(f'\tcalculated delta rho (full_out {full_out}, stats_out {stats_out}) and libsize aggregation ({aggregate_libsize_table}) {name}', file=sys.stdout, flush=True)
        log_line(logger, f'\tcalculated delta rho (full_out {full_out}, stats_out {stats_out}) and libsize aggregation ({aggregate_libsize_table}) {name}',
                 indent=0,
                 log_type="debug")
        output_collections.append(output_col)

    new_output_col = OutputCollection(in_table=output_collections, grp_specs=test_grp.get_group_config(), tmp_dir=tmp_dir)
    e_val = grp_info.get("E")
    tau_val = grp_info.get("tau")
    et_tag = f"E{e_val}_tau{tau_val}"

    if aggregate_libsize_table is True:
        libsize_aggregated_path = existing_output.libsize_aggregated.path if existing_output is not None else None
        if libsize_aggregated_path is not None and et_tag in str(libsize_aggregated_path):
            new_output_col.libsize_aggregated.path = libsize_aggregated_path

    if calc_delta_rho_table is True:
        delta_rho_path = existing_output.delta_rho_stats.path if existing_output is not None else None
        if delta_rho_path is not None and et_tag in str(delta_rho_path):
            new_output_col.delta_rho_stats.path = delta_rho_path

    if calc_delta_rho_full is True:
        delta_rho_path_full = existing_output.delta_rho_full.path if existing_output is not None else None
        if delta_rho_path_full is not None and et_tag in str(delta_rho_path_full):
            new_output_col.delta_rho_full.path = delta_rho_path_full

    if new_output_col.libsize_aggregated is None:
        if existing_output is not None:
            new_output_col.libsize_aggregated = existing_output.libsize_aggregated
            new_output_col.libsize_aggregated.get_table()

    if new_output_col.delta_rho_stats is None:
        if existing_output is not None:
            new_output_col.delta_rho_stats = existing_output.delta_rho_stats
            new_output_col.delta_rho_stats.get_table()

    if new_output_col.delta_rho_full is None:
        if existing_output is not None:
            new_output_col.delta_rho_full = existing_output.delta_rho_full
            new_output_col.delta_rho_full.get_table()

    # Safeguard: keep only rows whose relationship is between the two vars declared in proj_config.
    allowed_vars = {config.col.var, config.target.var}
    _apply_relationship_safeguard(new_output_col.delta_rho_stats, allowed_vars, "delta_rho_stats")
    _apply_relationship_safeguard(new_output_col.delta_rho_full, allowed_vars, "delta_rho_full")
    _apply_relationship_safeguard(new_output_col.libsize_aggregated, allowed_vars, "libsize_aggregated")

    try:
        gb = new_output_col.libsize_aggregated.surrogate.group_by(["surr_var"]).aggregate([("surr_num", "count_distinct")])
        df = gb.to_pandas()

        new_output_col.delta_rho_stats.write_table(tag=f"E{e_val}_tau{tau_val}__delta_rho_stats")
        # print('\twriting delta rho stats table', file=sys.stdout, flush=True)
        log_line(logger, '\twriting delta rho stats table',
                 indent=0,
                 log_type="info")

        new_output_col.delta_rho_full.write_table(tag=f"E{e_val}_tau{tau_val}__delta_rho_full")
        # print('\twriting delta rho full table', file=sys.stdout, flush=True)
        log_line(logger, '\twriting delta rho full table',
                 indent=0,
                 log_type="info")

        new_output_col.libsize_aggregated.write_table(tag=f"E{e_val}_tau{tau_val}__libsize_aggregated")
        # print('\twriting libsize aggregated table', file=sys.stdout, flush=True)
        log_line(logger, '\twriting libsize aggregated table',
                 indent=0,
                 log_type="info")

    except Exception as e:
        # print("Error pulling output for E={E}, tau={tau}: {error}".format(E=E, tau=tau, error=e))
        log_line(logger, "Error pulling output for E={E}, tau={tau}: {error}".format(E=E, tau=tau, error=e),
                 indent=0,
                 log_type="error")

    # print('\tclearing tables', file=sys.stdout, flush=True)
    log_line(logger, '\tclearing tables',
             indent=0,
             log_type="info")
    new_output_col.clear_tables()

    cell_obj = GridCell(E_i, tau_i, new_output_col)
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

    calc_location = set_calc_path(args, proj_dir, config)
    log_line(logger, f'Calculation location: {calc_location}', indent=0, log_type="info")
    # print(f'Calculation location: {calc_location}', file=sys.stdout, flush=True)
    # print(f'Read e_tau_grps_df from {group_file_name}.', file=sys.stdout, flush=True)
    log_line(logger, f'Read e_tau_grps_df from {group_file_name}.', indent=0, log_type="info")

    e_tau_grps_df = pd.read_csv(calc_location / check_csv(group_file_name))

    output_location = resolve_consolidated_dir(calc_location, config, "parquet")
    # output_parquet_location = output_location / output_config.dir_structure
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
    if not_in_grid is True or output_is_none is True:
        # print('regardless of flags, going the dual calculations', file=sys.stdout, flush=True)
        log_line(logger, 'regardless of flags, going the dual calculations', indent=0,
                 log_type="info")

        object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config, calc_delta_rho_table=True,
                                               aggregate_libsize_table=True, calc_delta_rho_full=True)

        joblib_cloud_atomic_dump(object_grid, tmp_dir / obj_grid_file_name, compress=3,
                                 protocol=5)
        del object_grid
        gc.collect()
        log_line(logger, f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", indent=0,
                 log_type="info")
        # print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
    else:
        if object_grid[(E, tau)].output is None:
            calc_delta_rho_table = True
            aggregate_libsize_table = True
            calc_delta_rho_table_full = True
            log_line(logger, 'output is None, going the dual calculations', indent=0,
                     log_type="info")
            # print('output is None, going the dual calculations', file=sys.stdout, flush=True)
        else:
            if (object_grid[(E, tau)].output.delta_rho_stats is None) or (object_grid[(E, tau)].output.delta_rho_stats.path is None):
                calc_delta_rho_table = True
            if (object_grid[(E, tau)].output.delta_rho_full is None) or (object_grid[(E, tau)].output.delta_rho_full.path is None):
                calc_delta_rho_table_full = True
                calc_delta_rho_table = True
            if (object_grid[(E, tau)].output.libsize_aggregated is None) or (object_grid[(E, tau)].output.libsize_aggregated.path is None):
                aggregate_libsize_table = True

        log_line(logger, ['calculations have been explicitly set: calc_delta_rho_table', calc_delta_rho_table,
              '; aggregate_libsize:', aggregate_libsize_table], indent=0,
                 log_type="info")
        # print('calculations have been explicitly set: calc_delta_rho_table', calc_delta_rho_table,
        #       '; aggregate_libsize:', aggregate_libsize_table, file=sys.stdout, flush=True)
        if (calc_delta_rho_table is True) or (aggregate_libsize_table is True) or (calc_delta_rho_table_full is True):

            object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config, existing_output=object_grid[(E, tau)].output,
                                                       calc_delta_rho_table=calc_delta_rho_table,
                                                       aggregate_libsize_table=aggregate_libsize_table,
                                                   calc_delta_rho_full = calc_delta_rho_table_full)

            joblib_cloud_atomic_dump(object_grid, tmp_dir/obj_grid_file_name, compress=3,
                                   protocol=5)
            del object_grid
            gc.collect()
            log_line(logger, f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", indent=0,
                     log_type="info")
            # print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
        else:
            log_line(logger, f"Skipping E={E}, tau={tau} because already processed.", indent=0,
                     log_type="info")
            # print(f"Skipping E={E}, tau={tau} because already processed.", file=sys.stdout, flush=True)

    # Process if either calculation flag is set
    # elif (calc_delta_rho_table is True) or (aggregate_libsize_table is True):
    #     print('calculations have been explicitly set: calc_delta_rho_table', calc_delta_rho_table,
    #           '; aggregate_libsize:', aggregate_libsize_table, file=sys.stdout, flush=True)
    #
    #     object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config, existing_output=object_grid[(E, tau)].output,
    #                                            calc_delta_rho_table=calc_delta_rho_table,
    #                                            aggregate_libsize_table=aggregate_libsize_table)
    #
    #     joblib_cloud_atomic_dump(object_grid, tmp_dir/obj_grid_file_name, compress=3,
    #                        protocol=5)
    #     del object_grid
    #     gc.collect()
    #     print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
    # else:
    #     print(f"Skipping E={E}, tau={tau} because already processed.", file=sys.stdout, flush=True)
