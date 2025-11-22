import time

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

import cedarkit.utils.tables.parquet_tools

# python
# Preferred: absolute imports (works if you run as: python -m cedarkit.local2.calc_grps)
try:
    from cedarkit.utils.cli.arg_parser import get_parser
    from cedarkit.core.project_config import load_config
    # adjust the module path below to where these helpers actually live in your project
    from cedarkit.utils.routing.paths import check_location, set_calc_path, set_output_path
    from cedarkit.utils.cli.arg_parser import get_parser
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.arg_parser import get_parser
    from core.project_config import load_config
    from utils.paths import check_location, set_calc_path, set_output_path

'''
Main script to assign group IDs based on unique parameter combinations.
Groups are deduplicated based on specified query keys from the configuration.
Each E-tau-lag-col_var_id-target_var_id combination constitutes.
'''

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.project is None:
        print('Project name is required.', file=sys.stderr)
        sys.exit(1)

    proj_name = args.project
    proj_dir = Path(os.getcwd()) / proj_name

    # Load main config
    config_path = proj_dir / f'{args.config or "proj_config"}.yaml'
    config = load_config(config_path)

    # Determine location where script is being run and load location-specific config
    loc_config = config.get_dynamic_attr('{var}', check_location(proj_dir))

    calc_location = set_calc_path(args, proj_dir, config)
    calc_location.mkdir(parents=True, exist_ok=True)

    output_dir = set_output_path(args, calc_location, config)

    query_keys = config.calc_grps.query_keys
    parameter_dir = proj_dir / loc_config.parameter_dir

    # Load parameters
    if args.parameters not in (None, ''):
        params_df = pd.read_csv(parameter_dir / f'{args.parameters}.csv')
    else:
        try:
            parameters_files = [fl for fl in os.listdir(str(parameter_dir)) if (('.csv' in fl) and ('_bycol_' in fl) and not fl.startswith('summary_')) ]#config.parameters if isinstance(config.parameters, list) else [config.parameters]
            print(parameters_files)
            param_dfs = [pd.read_csv(parameter_dir / f'{fname}') for fname in parameters_files]
            params_df = pd.concat(param_dfs, ignore_index=True)
        except Exception as e:
            print(f'Parameter file is required. ({e})', file=sys.stderr)
            sys.exit(1)

    # Handle slice
    start_ind = 0 if args.inds is None else args.inds[0]
    end_ind = None if args.inds is None else args.inds[1]
    params_df = params_df.iloc[start_ind:end_ind]

    # Deduplicate groups
    grouped = params_df.drop_duplicates(ignore_index=True)

    # Paths to input/output CSVs
    grp_csv_base = config.csvs.calc_grps
    grp_csv_path = calc_location / f'{grp_csv_base}.csv'

    # Apply timestamp suffix if running in test mode
    second_suffix = f'_{int(time.time() * 1000)}' if args.test else ''
    grp_csv_path_out = calc_location / f'{grp_csv_base}{second_suffix}.csv'
    # grp_run_csv_path_out = calc_location / f'{grp_run_csv_base}{second_suffix}.csv'

    # GROUP ASSIGNMENT WORKFLOW
    file_written = False
    if grp_csv_path.exists() is True:
        try:
            grouped_df = pd.read_csv(grp_csv_path)
            print(f"Loaded existing group assignments from {grp_csv_path}.", file= sys.stdout, flush=True)
            grouped_df['group_id'] = grouped_df['group_id'].astype(int)
            grouped_df['train_ind_i'] = grouped_df.get('train_ind_i', 0).astype(int)

            merged = grouped.merge(grouped_df, how='outer', on=query_keys)
            missing = merged[merged['group_id'].isna()].copy()

            if len(missing) > 0:
                start_time = int(time.time())
                missing['group_id'] = np.arange(start_time, start_time + len(missing)).astype(int)
                merged = pd.concat([merged, missing], ignore_index=True)

                grouped_df = merged.dropna(subset=['group_id']).drop_duplicates(ignore_index=True)
                grouped_df['group_id'] = grouped_df['group_id'].astype(int)
                grouped_df.to_csv(grp_csv_path_out, index=False)
                print(f"Assigned group_id to {len(missing)} new rows.", flush=True)

            file_written = True
        except Exception as e:
            print(f"Error reading existing group assignments: {e}", flush=True)

    if file_written is False:
        print(f"No valid existing group assignments found. Creating new group assignments.", flush=True)
        # Fresh group_id assignment
        start_time = int(time.time())
        grouped['group_id'] = np.arange(start_time, start_time + len(grouped))
        grouped_df = grouped.copy()
        grouped_df.to_csv(grp_csv_path_out, index=False)

    grouped_df[['col_var_id','target_var_id','E','tau','knn','Tp']].drop_duplicates(ignore_index=True).to_csv(calc_location / 'E_tau_grps.csv', index=False)
    print('Group assignments processed successfully!', flush=True)


if __name__ == '__main__':
    main()





