import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import os, sys

from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.data_access import collect_raw_data, get_weighted_flag, set_df_weighted, write_query_string

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
def get_real_data(real_dfs_references, meta_variables, max_libsize, knn, grp_id, grp_path, sample_size=500):
    surr_var = 'neither'
    surr_num = 0
    pset_id = real_dfs_references['pset_id'].unique()[0]

    E = real_dfs_references['E'].unique()[0]
    tau = real_dfs_references['tau'].unique()[0]

    df_csv_name = f'{pset_id}_E{E}_tau{tau}__{surr_var}{surr_num}.csv'
    df_csv_path = grp_path / df_csv_name

    if df_csv_path.exists() == True:
        print(f'[real exists] real output grp_id:{grp_id}, pset_id:{pset_id} already exists, {df_csv_name}', file=sys.stdout, flush=True)
        return None

    real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)
    if real_df_full.empty:
        return None

    real_df_full['surr_var'] = surr_var
    real_df_full['surr_num'] = surr_num
    real_df_full['relation_s'] = real_df_full['relation']

    real_df_full.to_csv(df_csv_path, index=False)


# Function to fetch and prepare surrogate data
# Goal: save a single csv for each surr pset_id group with name {grp_id}_{surr_var}_{surr_num}.csv
def get_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, grp_id,grp_path,
                       sample_size=400):

    for pset_id, pset_df in surr_dfs_references.groupby('pset_id'):
        E = pset_df['E'].unique()[0]
        tau = pset_df['tau'].unique()[0]
        surr_var = pset_df['surr_var'].unique()[0]
        surr_num = pset_df['surr_num'].unique()[0]

        df_csv_name = f'{pset_id}_E{E}_tau{tau}__{surr_var}{surr_num}.csv'
        df_csv_path = grp_path / df_csv_name

        if df_csv_path.exists():
            print(f'[surrogate exists] surrogate output grp_id:{grp_id}, pset_id:{pset_id} already exists, {df_csv_name}', file=sys.stdout, flush=True)
            continue

        surr_df = collect_raw_data(pset_df, meta_vars=meta_variables)
        surr_df = surr_df[surr_df['surr_var'] != 'neither'].copy()
        surr_df['surr_num'] = surr_num

        if surr_df.empty:
            print(f'no surrogate data found for {grp_id}, E={E}, tau={tau}', grp_path, file=sys.stdout, flush=True)
            continue

        rel_dfs = []
        for surr_var, surr_df_i in surr_df.groupby('surr_var'):
            surr_df_i['relation_s'] = surr_df_i['relation'].str.replace(surr_var, f'{surr_var} (surr) ')
            surr_df_i['relation_s'] =surr_df_i['relation_s'].str.strip()
            surr_df_i['relation_s'] = surr_df_i['relation_s'].str.replace('  ', ' ')
            rel_dfs.append(surr_df_i)
        surr_df = pd.concat(rel_dfs).reset_index(drop=True)
        if len(surr_df) == 0:
            print(f'[stage 2] no surrogate data found for {grp_id}, E={E}, tau={tau}', grp_path, file=sys.stdout, flush=True)
            continue

        try:
            for (surr_num, surr_var), surr_grp in surr_df.groupby(['surr_num', 'surr_var']):
                df_csv_name = f'{pset_id}_E{E}_tau{tau}__{surr_var}{surr_num}.csv'
                df_csv_path = grp_path / df_csv_name
                surr_df.to_csv(df_csv_path, index=False)
                print(f'Writing surrogate data to {df_csv_path}', file=sys.stdout, flush=True)
        except:
            print(f'Error writing surrogate data: {grp_path}, pset_id: {pset_id}, len={len(surr_df)}',  file=sys.stdout, flush=True)
            continue

        # save csv


def process_group_workflow(arg_tuple):
    (grp_d, ind, real_dfs_references, surr_dfs_references, pctile_range, override, write,  config, calc_location) = arg_tuple
    print('E', grp_d['E'], 'tau', grp_d['tau'], 'ind', ind, 'group_id', grp_d['group_id'], file=sys.stdout, flush=True)
    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']

    max_libsize = 325
    knn = 20

    grp_path = calc_location / 'calc_refactor'/f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'/ f'E{grp_d["E"]}_tau{grp_d["tau"]}'
    grp_path.mkdir(exist_ok=True, parents=True)

    get_real_data(real_dfs_references, meta_variables, max_libsize, knn, grp_d['group_id'], grp_path)
    get_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, grp_d['group_id'], grp_path)


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

    override = args.override  # if args.override not in [None else False
    write_flag = args.write
    if override is False:
        if write_flag in ['append', None]:
            write_flag = 'append'

    second_suffix = ''

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    calc_metric_dir_name = carc_config_d.dirs.calc_metrics_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  # 'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

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

    if Path('/Users/jlanders').exists():

        calc_grps_df = calc_grps_df[(calc_grps_df['train_ind_i'] == 0) &
                                    (calc_grps_df['lag'] == 0) &
                                    (calc_grps_df['knn'] == 20)
        ].copy()
        calc_grps = calc_grps_df.to_dict(orient='records')

        arg_tuples = []
        ind = 0
        for grp_d in calc_grps:

            grp_df = calc_log2_df.query(write_query_string(query_keys, grp_d))
            weighted_flag = get_weighted_flag(grp_d)
            grp_df = set_df_weighted(grp_df, weighted_flag)

            # Filter for real data frames
            real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
            surr_dfs_references = grp_df[grp_df['surr_var'] != 'neither'].copy()

            arg_tuples.append((grp_d, ind, real_dfs_references, surr_dfs_references, pctile_range,
                                       override, write_flag, config, calc_location))

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

        grp_ds = calc_grps_df.to_dict(orient='records')
        if index >= len(grp_ds):
            print('index out of range', file=sys.stdout, flush=True)
            sys.exit(1)
        else:
            grp_d = grp_ds[index]

        grp_df = calc_log2_df.query(write_query_string(query_keys, grp_d))
        weighted_flag = get_weighted_flag(grp_d)
        grp_df = set_df_weighted(grp_df, weighted_flag)

        # Filter for real data frames
        real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
        surr_dfs_references = grp_df[grp_df['surr_var'] != 'neither'].copy()

        arg_tuple = (grp_d, index, real_dfs_references, surr_dfs_references, pctile_range,
                                       override, write_flag, config, calc_location)

        process_group_workflow(arg_tuple)
