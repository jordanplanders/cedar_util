from operator import index
from pathlib import Path
import os
import sys
import gc
import numpy as np
import pandas as pd
pd.option_context('mode.use_inf_as_na', True)

from utils.config_parser import load_config
from utils.arg_parser import get_parser, parse_flags
from utils.location_helpers import *
from utils.data_access import *
from data_obj.data_objects import *#DataGroup, GroupOutput
from data_obj.plotting_objects import *


def process_config(grp_info, E_i, tau_i, tmp_dir, output_location, config):

    test_grp = DataGroup(grp_info, tmp_dir=tmp_dir)
    test_grp.get_files(config, output_location / 'parquet',
                       file_name_pattern='E{E}_tau{tau}_lag{lag}', source='parquet')

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
        output_col = groupconfig_file.pull_output(to_table=False).calc_delta_rho().aggregate_libsize()
        print(f'\tcalculated delta rho and libsize aggregation {name}', file=sys.stdout, flush=True)

        output_col.table.clear_table()
        print('\tcleared table from memory', file=sys.stdout, flush=True)
        output_collections.append(output_col)

    new_output_col = OutputCollection(in_table=output_collections, grp_specs=test_grp.get_group_config(), tmp_dir=tmp_dir)

    try:
        gb = new_output_col.libsize_aggregated.surrogate.group_by(["surr_var"]).aggregate([("surr_num", "count_distinct")])
        df = gb.to_pandas()

        new_output_col.delta_rho_stats.write_table()
        print('\twriting delta rho stats table', file=sys.stdout, flush=True)

        new_output_col.libsize_aggregated.write_table()
        print('\twriting libsize aggregated table', file=sys.stdout, flush=True)

    except Exception as e:
        print("Error pulling output for E={E}, tau={tau}: {error}".format(E=E, tau=tau, error=e))

    print('\tclearing tables', file=sys.stdout, flush=True)
    new_output_col.clear_tables()

    cell_obj = GridCell(E_i, tau_i, new_output_col)
    del new_output_col

    cell_obj.row_labels.append(f'E={E}')
    cell_obj.col_labels.append(f'tau={tau}')

    for _, row in df.iterrows():
        cell_obj.annotations.append(f"{row['surr_var']}: n={row['surr_num_count_distinct']}")

    cell_obj.occupied = True
    return cell_obj


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # When run from the command line, assumes that the current working directory is the directory containing the proj_name (dyad) directory e.g. hol_temp_tsi_ccm
    proj_dir = Path(os.getcwd()) / proj_name
    gen_config = 'proj_config'
    config = load_config(proj_dir / f'{gen_config}.yaml')

    obj_grid_file_name = args.file if args.file is not None else f'{proj_name}_wu_obj_grid.joblib'
    group_file_name = args.group_file if args.group_file is not None else config.csvs.e_tau_grps
    tmp_dir = args.dir if args.dir is not None else'tmp' #target directory for cell object files and object_grid

    if args.inds is not None:
        ind = int(args.inds[-1])
    else:
        ind = int(sys.argv[-1])

    calc_location = set_calc_path(None, proj_dir, config, '')
    e_tau_grps_df = pd.read_csv(calc_location / check_csv(group_file_name))

    output_location = set_output_path(None, calc_location, config)
    tmp_dir = proj_dir / tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # this is hardcoded but should be released and left to the construction of the e_tau_grps_df
    E_vals = [4, 5, 6, 7, 8, 9]
    E_vals = [4, 5, 6, 7, 8, 9]
    tau_vals = [1, 2, 3, 4, 5, 6, 7]
    comb_df = e_tau_grps_df[e_tau_grps_df['E'].isin(E_vals) & e_tau_grps_df['tau'].isin(tau_vals)].copy()
    comb_plot_df = comb_df[[col for col in comb_df.columns if col != 'lag']].drop_duplicates()
    comb_plot_df = comb_plot_df.sort_values(by=['col_var_id', 'target_var_id', 'E', 'tau'])

    row = comb_plot_df.iloc[ind].to_dict()
    E = row['E']
    tau = row['tau']

    E_is = {E: ik for ik, E in enumerate(np.arange(min(E_vals), max(E_vals) + 1))}
    tau_is = {tau: ik for ik, tau in enumerate(np.arange(min(tau_vals), max(tau_vals) + 1))}

    try:
        object_grid = joblib_safe_load(tmp_dir / obj_grid_file_name, mmap_mode=None)
    except:
        object_grid = {}

    if (E, tau) not in object_grid:
        object_grid[(E, tau)] = process_config(row, E_is[E], tau_is[tau], tmp_dir, output_location, config)
        joblib_atomic_dump(object_grid, tmp_dir/obj_grid_file_name, compress=3,
                           protocol=5)
        del object_grid
        gc.collect()
        print(f"Processed and saved E={E}, tau={tau} to {tmp_dir}.", file=sys.stdout, flush=True)
    else:
        print(f"Skipping E={E}, tau={tau} because already processed.", file=sys.stdout, flush=True)


