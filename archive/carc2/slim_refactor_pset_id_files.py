# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from scipy import stats
# import datetime

import pandas as pd
from multiprocessing import Pool
import re
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.data_access import collect_raw_output, get_weighted_flag
# from utils.data_access import pull_percentile_data, get_group_sizes, get_sample_rep_n, check_empty_concat, get_surrogate_data, get_real_data
from utils.location_helpers import *


def streamline_cause(label, split_word='causes'):
    label_parts = label.split(' {} '.format(split_word))
    clean_parts = []
    for part in label_parts:
        # print(part.split('(t-'))
        try:
            part = ' '.join(list(set([col.split(')')[-1].strip(' ') for col in part.split('(t-')]))).strip(' ')
        except:
            part = part.strip(' ')
        clean_parts.append(part)
    label = f' {split_word} '.join(clean_parts)
    return label

############ Data Grab
# Function to fetch and prepare real data
def slim_real_data(real_dfs_references, meta_variables, max_libsize, knn, grp_id, grp_path, sample_size=500):

    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len', 'train_ind_i']
    files = real_dfs_references
    if isinstance(real_dfs_references,list) ==True:
        if len(real_dfs_references) == 0:
            print('No real data found', file=sys.stderr, flush=True)
            return None
        elif len(real_dfs_references) > 1:
            # files = [real_dfs_references[0]]
            real_df_full = pd.concat([pd.read_csv(grp_path / file) for file in files])
            print('Multiple files found, concatenating:', len(files), file=sys.stderr, flush=True)
        else:
            # files = real_dfs_references
            real_df_full = pd.read_csv(grp_path / files[0])
        # print(real_df_full.head(), file=sys.stderr, flush=True)
    else:
        real_df_full = collect_raw_output(real_dfs_references, meta_vars=meta_variables)
    # real_df_full = pd.read_csv(grp_path / files[0])

    real_df = real_df_full[[col for col in real_df_full.columns if col not in remove_cols]].copy()
    real_df['surr_var'] = 'neither'
    real_df['surr_num'] = 0

    # return real_df, grp_path/real_dfs_references[0]
    df_csv_name = files[0].split('__')[0] + '__' + 'neither' + '.csv'
    df_csv_path = grp_path / df_csv_name
    real_df = real_df.drop_duplicates(ignore_index=True)
    real_df.to_csv(df_csv_path, index=False)
    print('Real data saved:', df_csv_path, len(real_df), file=sys.stdout, flush=True)


# Function to fetch and prepare surrogate data
# Goal: save a single csv for each surr pset_id group with name {grp_id}_{surr_var}_{surr_num}.csv
def slim_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, grp_id,grp_path,
                       sample_size=400):

    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len',
                   'train_ind_i']

    surr_dfs = []
    for df_csv_name in surr_dfs_references:
        surr_df_full = pd.read_csv(grp_path / df_csv_name)
        file_name_parts = df_csv_name.split('__')
        if 'temp' in file_name_parts[-1]:
            surr_var = 'temp'
        elif 'tsi' in file_name_parts[-1].lower():
            surr_var = 'TSI'

        surr_df = surr_df_full[[col for col in surr_df_full.columns if col not in remove_cols]].copy()
        surr_df['surr_var'] = surr_var

        file_name_parts = df_csv_name.split('__')
        surr_num = int(re.search(r'\d+', file_name_parts[-1]).group())
        surr_df['surr_num'] = surr_num

        df_csv_path = grp_path / df_csv_name
        surr_df.to_csv(df_csv_path, index=False)

    master_surr_df = pd.concat(surr_dfs)
    df_csv_name = df_csv_name.split('__')[0]+'__'+surr_var+'.csv'
    df_csv_path = grp_path / df_csv_name
    master_surr_df.to_csv(df_csv_path, index=False)
    print('Surrogate data saved:', df_csv_path,len(master_surr_df), file=sys.stdout, flush=True)



def process_group_workflow(arg_tuple):
    (grp_d, ind, real_dfs_references, surr_dfs_references, pctile_range, args,  config, calc_location, output_dir) = arg_tuple
    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']

    max_libsize = 325
    knn = 20

    grp_path = output_dir/f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'/ f'E{grp_d["E"]}_tau{grp_d["tau"]}'
    grp_path.mkdir(exist_ok=True, parents=True)

    slim_real_data(real_dfs_references, meta_variables, max_libsize, knn, grp_d['group_id'], grp_path)
    slim_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, grp_d['group_id'], grp_path)
    print('E', grp_d['E'], 'tau', grp_d['tau'], 'ind', ind, 'group_id', grp_d['group_id'], file=sys.stdout, flush=True)


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # override = args.override  # if args.override not in [None else False
    # write_flag = args.write
    # if override is False:
    #     if write_flag in ['append', None]:
    #         write_flag = 'append'

    second_suffix = ''


    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    calc_metric_dir_name = carc_config_d.dirs.calc_metrics_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    # if Path('/Users/jlanders').exists() == True:
    #     calc_location = proj_dir / config.local.calc_carc  # 'calc_local_tmp'
    # else:
    #     calc_location = proj_dir / config.carc.calc_carc
    calc_location = set_calc_path(args, proj_dir, config)
    output_dir = set_output_path(args, calc_location, config)

    if args.group_file is not None:
        grp_csv = f'{args.group_file}.csv'
    else:
        grp_csv = f'{carc_config_d.csvs.calc_grp_run_csv}.csv'

    calc_grps_path = calc_carc_mirrored / grp_csv

    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()
    calc_grps_df = calc_grps_df[calc_grps_df['Tp'] == 20].copy()

    pctile_range = [.25, .75]
    query_keys = config.calc_criteria_rates2.query_keys

    # figs_dir.mkdir(exist_ok=True, parents=True)

    # if len(sys.argv) > 1:
    #     delta_label = sys.argv[1]
    # else:
    # delta_label = 'r50p-s50p'

    # calc_grps_path = calc_carc / 'calc_grps.csv'
    # calc_grps_df = pd.read_csv(calc_grps_path)
    # calc_grps_df = calc_grps_df[(calc_grps_df['tau'].isin([8]))].copy()
    # # calc_grps_df = calc_grps_df[(calc_grps_df['tau']==6) &(calc_grps_df['E']==9)].copy()
    # calc_grps = [(grp_d, delta_label) for grp_d in calc_grps_df.to_dict(orient='records')]
    #
    # # Determine the number of CPUs to use from the SLURM environment variable
    # num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 4))
    #
    # # Use multiprocessing to parallelize the process
    # with Pool(num_cpus) as pool:
    #     results = pool.map(process_group, calc_grps)
    #
    # print('figures built successfully!', file = sys.stdout, flush = True)
    #
    #
    # calc_grps_path = calc_carc / 'calc_grps.csv'
    # calc_grps_df = pd.read_csv(calc_grps_path)
    # calc_grps_df = calc_grps_df[(calc_grps_df['tau'].isin([8]))].copy()
    # calc_grps_df = calc_grps_df[(calc_grps_df['tau']==4)].copy()

    ## scratch code for checking dates
    # if metric_df_path.exists():
    #     # file modification timestamp of a file
    #     m_time = os.path.getmtime(metric_df_path)
    #     # convert timestamp into DateTime object
    #     dt_m = datetime.datetime.fromtimestamp(m_time)
    #     if dt_m > datetime.datetime(2024, 8, 12):
    #         print(f'grp_id:{str(grp_d["group_id"])} already exists', file=sys.stdout, flush=True)
    if Path('/Users/jlanders').exists():

        calc_grps_df = calc_grps_df[(calc_grps_df['train_ind_i'] == 0) &
                                    (calc_grps_df['lag'] == 0) &
                                    (calc_grps_df['knn'] == 20)
                                    # & (calc_grps_df['col_var_id'] == 'essel')
                                    # & (calc_grps_df['target_var_id'] == 'vieira')
                                    # & (calc_grps_df['E'] == 4)
                                   # & (calc_grps_df['tau'] < 8)
        ].copy()
        calc_grps = calc_grps_df.to_dict(orient='records')

        arg_tuples = []
        ind = 0
        for grp_d in calc_grps:
            # conv_match = convergence_grps_df[convergence_grps_df['group_id'] == grp_d['group_id']].copy()
            # if len(conv_match) > 0:
            #     conv_match_d = conv_match.iloc[0].to_dict()
            #     if len(conv_match_d)>0:
            # grp_df = calc_log2_df.query(write_query_string(query_keys, grp_d))
            grp_path = output_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'E{grp_d["E"]}_tau{grp_d["tau"]}'
            weighted_flag = get_weighted_flag(grp_d)

            files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
            real_dfs_references = [file for file in files if 'neither' in file]
            surr_dfs_references = [file for file in files if 'neither' not in file]

            # grp_df = set_df_weighted(grp_df, weighted_flag)
            #
            # # Filter for real data frames
            # real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
            # surr_dfs_references = grp_df[grp_df['surr_var'] != 'neither'].copy()

            arg_tuples.append((grp_d, ind, real_dfs_references, surr_dfs_references, pctile_range,
                                       args, config, calc_location, output_dir))

            ind += 1
        num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([1, len(arg_tuples)])))

        # Use multiprocessing to parallelize the process
        with Pool(num_cpus) as pool:
            pool.map(process_group_workflow, arg_tuples)


    else:
        if args.inds is not None:
            index = int(args.inds[-1])
        else:
            print('calc_criteria_rates2; index is required', file=sys.stdout, flush=True)
            sys.exit(0)
        # index = int(sys.argv[1])

        grp_ds = calc_grps_df.to_dict(orient='records')
        if index >= len(grp_ds):
            print('index out of range', file=sys.stdout, flush=True)
            sys.exit(1)
        else:
            grp_d = grp_ds[index]

        # grp_df = calc_log2_df.query(write_query_string(query_keys, grp_d))
        # weighted_flag = get_weighted_flag(grp_d)
        # grp_df = set_df_weighted(grp_df, weighted_flag)
        #
        # # Filter for real data frames
        # real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
        # surr_dfs_references = grp_df[grp_df['surr_var'] != 'neither'].copy()

        grp_path = output_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'E{grp_d["E"]}_tau{grp_d["tau"]}'
        files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
        real_dfs_references = [file for file in files if 'neither' in file]
        surr_dfs_references = [file for file in files if 'neither' not in file]

        arg_tuple = (grp_d, index, real_dfs_references, surr_dfs_references, pctile_range,
                                       args, config, calc_location, output_dir)

        process_group_workflow(arg_tuple)
