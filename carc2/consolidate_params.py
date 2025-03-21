import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import os
import datetime
import sys
import yaml
from utils.arg_parser import get_parser

# Paths to the directories and files
# notebook_path = Path(os.getcwd())/'notebooks'
# print(notebook_path, file=sys.stdout, flush=True)
# calc_carc = notebook_path/'calc_carc'
# calc_log_path = calc_carc/'calc_log2.csv'
# calc_log_path_new = calc_carc/'calc_log4.csv'
# missing_params =calc_carc/"missing_params_new.csv"

# calc_dir = notebook_path/'calc_dir'

def process_group(arg_tuple):
    grp, grp_df, calc_dir, proj_name = arg_tuple
    files = [file for file in os.listdir(calc_dir/str(grp)) if ('pctile' not in file) & ('.csv' in file) & ('df' in file)]
    grps = []

    proj_ind = [ik for ik,_ in enumerate(calc_dir.parts) if _ == proj_name][0]
    path_parts = ['.']+list(calc_dir.parts[proj_ind:]) + [str(grp)]
    # relative_path = calc_dir.relative_to(proj_name)
    path_to_record = '/'.join(path_parts)

    # if calc_dir.parents[0].name == 'carc':
    for file in files:
        _grp_df = grp_df.copy()
        run_id = file.replace('df_', '').replace('.csv', '')
        print('run_id',run_id, file, file=sys.stdout, flush=True)

        _grp_df['df_path'] = f'{str(path_to_record)}/{file}'#f'./notebooks/{calc_dir_name}/'+str(grp)+'/'+file
        # _grp_df['df_path'] = f'./notebooks/{calc_dir_name}/'+str(grp)+'/'+file

        _grp_df['run_id'] = run_id
        grps.append(_grp_df)
    # grps.append(_grp_df)
    return grps


def extract_missing_calc(pset_dirs, param_df):
    existing_indexes = [index for index in param_df.index if index in pset_dirs]
    calc_dir_data_runs = param_df.loc[existing_indexes, :].copy()

    missing_indexes = [index for index in param_df.index if index not in pset_dirs]
    missing_runs = param_df.loc[missing_indexes, :].copy()

    print(f'len missing: {len(missing_runs)}, len total: {len(param_df)}', file=sys.stdout, flush=True)
    return calc_dir_data_runs, missing_runs


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    # print(f"Number from script2: {args.number}")
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)
        # print("Script2 in verbose mode")
    if args.parameters  is not None:
        parameter_flag = args.parameters
    else:
        print('parameters are required', file=sys.stdout, flush=True)
        print('parameters are required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    with open(proj_dir / 'proj_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    carc_config_d = config["calc_carc_dir"]
    calc_carc = proj_dir/carc_config_d["name"]

    calc_dir = proj_dir/config["calc_dir"]

    start_ts = datetime.datetime.now()
    parameter_dir = proj_dir / 'parameters'
    parameter_path = parameter_dir / f'{parameter_flag}.csv'

    start_ind = 0
    end_ind = -1
    if args.inds is not None:
        start_ind = int(args.inds[-1])
        if len(args.inds) > 1:
            end_ind = int(args.inds[-2])


    # calc_log_path = calc_carc / 'calc_log2.csv'
    calc_log_path_new = calc_carc / f'{carc_config_d["csvs"]["completed_runs_csv"]}.csv'
    missing_params = calc_carc / f'{carc_config_d["csvs"]["missing_runs_csv"]}.csv'


    # calc_dir_data = pd.read_csv(calc_carc/'pset_run_files.csv')#, index_col=0)
    _pset_dirs = pd.read_csv(calc_carc/f'{carc_config_d["csvs"]["pset_dirs_list"]}.txt', header=None)#, index_col=0)
    # _pset_dirs = pd.read_csv(calc_carc/f'pset_dirs.txt', header=None)#, index_col=0)

    pset_dirs = []
    for val in _pset_dirs.values.flatten():
        try:
            pset_dirs.append(int(val))
        except:
            continue


    #@todo: add support for multiple parameter files by changing flag to a list of parameter files
    total_missing = []
    calc_runs = []
    param_df = pd.read_csv(parameter_path, index_col=0)
    param_df = param_df.iloc[start_ind:end_ind, :]
    calc_dir_data_runs, missing_runs = extract_missing_calc(pset_dirs, param_df)
    calc_runs.append(calc_dir_data_runs)
    total_missing.append(missing_runs)
    # else:
    #     parameter_files = [
    #     "parameter_combinations-erb_vieira2.csv",
    #     "parameter_combinations-erb_wu2.csv",
    #     "parameter_combinations-essel_vieira2.csv",
    #     "parameter_combinations-essel_wu2.csv"
    #     ]
    #
    #     total_missing = []
    #     calc_runs = []
    #     for param_file in parameter_files:
    #         temp_author, tsi_author = param_file.split('-')[1].split('.')[0].split('_')
    #         temp_author = temp_author.replace('2', '')
    #         tsi_author = tsi_author.replace('2', '')
    #         param_df = pd.read_csv(parameter_dir/param_file, index_col=0)
    #         param_df['temp_author'] = temp_author
    #         param_df['tsi_author'] = tsi_author
    #         # calc_dir_data_runs = calc_dir_data.merge(param_df, left_on='pset_id', right_on='id', suffixes=('', '_param'))
    #         calc_dir_data_runs, missing_runs = extract_missing_calc(pset_dirs, param_df)
    #         calc_runs.append(calc_dir_data_runs)
    #         total_missing.append(missing_runs)

        # existing_indexes = [index for index in param_df.index if index in pset_dirs]
        # calc_dir_data_runs = param_df.loc[existing_indexes, :].copy()
        #
        # missing_indexes = [index for index in param_df.index if index not in pset_dirs]
        # missing_runs = param_df.loc[missing_indexes, :].copy()
        #
        # calc_runs.append(calc_dir_data_runs)
        # total_missing.append(missing_runs)
        # print(temp_author, tsi_author, len(missing_runs), len(param_df))

    calc_runs_df = pd.concat(calc_runs)
    total_missing_df = pd.concat(total_missing)
    calc_runs_df = calc_runs_df.reset_index()

    arg_tuples = [(grp, grp_df, calc_dir, proj_name) for grp, grp_df in calc_runs_df.groupby('id')]

    # Use multiprocessing to parallelize the process
    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    with Pool(num_cpus) as pool:
        results = pool.map(process_group, arg_tuples)

    # Collect results from parallel processing
    df_pathed_dfs = []
    for result_lst in results:
        df_pathed_dfs.extend(result_lst)

    calc_runs_df = pd.concat(df_pathed_dfs)
    calc_runs_df = calc_runs_df.rename(columns={'id': 'pset_id'})
    calc_runs_df = calc_runs_df.reset_index(drop=True)

    calc_runs_df.to_csv(calc_log_path_new)
    total_missing_df.to_csv(missing_params)

    end_ts = datetime.datetime.now()
    print(calc_log_path_new, file=sys.stdout, flush=True)
    print('end:', end_ts, 'elapsed:', end_ts - start_ts, file=sys.stdout, flush=True)





