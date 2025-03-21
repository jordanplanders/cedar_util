import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import sys
import datetime
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import time
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.data_access import collect_raw_data, get_weighted_flag, write_query_string , set_df_weighted
 # was calc_convergence_testing_approach2.py



def process_group(args):
    grp_d, config, calc_metric_dir, real_dfs_references, plot_flag, override_flag, datetime_flag = args

    target_relation = None
    # if config.has_nested_attribute('calc_criteria_rates2.target_relation'):
    #     target_relation = config.calc_criteria_rates2.target_relation

    start_time = datetime.datetime.now()

    if grp_d['weighted'] in ['True', True]:
        weighted_flag = 'weighted'
    elif grp_d['weighted'] in ['False', False]:
        weighted_flag = 'unweighted'

    ## Processing
    real_df = collect_raw_data(real_dfs_references)

    lag_ds = []
    for traits,  rel_lag_df in real_df.groupby(['relation', 'lag']):
        rel, lag = traits
        try:
            rel_lag_df[['rho', 'LibSize']] = rel_lag_df[['rho', 'LibSize']].astype(float)
            agg_rho_libsize = rel_lag_df[['rho', 'LibSize']].groupby('LibSize').mean().reset_index()
        except:
            print('error with agg_rho_libsize', grp_d, rel, rel_lag_df[['rho', 'LibSize']].head(), file=sys.stdout,
                  flush=True)
            return

        # Find the index of the maximum 'rho' value in the aggregated DataFrame
        iq = np.argmax(agg_rho_libsize['rho'])
        # Extract the 'LibSize' corresponding to the maximum 'rho' value
        max_rho_libsize = agg_rho_libsize['LibSize'].iloc[iq]


        max_libsize = agg_rho_libsize['LibSize'].max()
        final_rho = agg_rho_libsize[agg_rho_libsize['LibSize']==max_libsize]['rho'].values[0]

        grp_d_copy = grp_d.copy()
        grp_d_copy.update({'relation': rel, 'lag': lag, 'rho': agg_rho_libsize['rho'].max(), 'LibSize': max_rho_libsize, 'LibSize_type':'max_rho'})
        lag_ds.append(grp_d_copy)

        grp_d_copy = grp_d.copy()
        grp_d_copy.update({'relation': rel, 'lag': lag, 'rho': final_rho, 'LibSize': max_libsize, 'LibSize_type':'max'})
        lag_ds.append(grp_d_copy)

    lag_df = pd.DataFrame(lag_ds)
    return lag_df





# group by E, tau, Tp, knn, weighted, target_var, col_var
## could do this with percentile or raw, here we'll do it with raw

# group by lag, relationship
# group by LibSize, take mean rho
# collect max rho and final rho ---> dict(relation=relation, E=E, tau=tau, lag=lag, Tp=Tp, knn=knn, weighted=weighted, target_var=target_var, col_var=col_var, rho=rho, LibSize=LibSize, LibSize_type=LibSize_type)
# make dataframe
# save to csv

# group by relation, LibSize_type
# identify lag associated with max rho
# save to csv

## Plotting



# Is there a change in mean lib index and std deviation of lib index between max rho libsize and max libsize
# compare distribution of lib index between max rho libsize and max libsize
# talking about sample number x libSize arrays


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)


    if args.override:
        override = args.override
    else:
        override = False

    second_suffix = ''
    if args.test:
        # if sys.argv[-1] == 'test':
        second_suffix = f'_{int(time.time() * 1000)}'

    proj_dir = Path(os.getcwd()) / proj_name
    # with open(proj_dir / 'proj_config.yaml', 'r') as file:
        # config = yaml.safe_load(file)
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    calc_metric_dir_name = carc_config_d.dirs.lag_processing_dir  #config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    query_keys = config.lag_processing.query_keys

    calc_grps_path = calc_carc_mirrored / f'{carc_config_d.csvs.calc_grp_run_csv}.csv'
    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()

    calc_metric_dir = calc_location / calc_metric_dir_name
    calc_metric_dir.mkdir(parents=True, exist_ok=True)

    plot_flag = True
    # override_flag= False
    datetime_flag = None

    if Path('/Users/jlanders').exists():
        print('running locally', file=sys.stdout, flush=True)
        # calc_grps_df = calc_grps_df[(calc_grps_df['tau'] == 4) &
        #                             (calc_grps_df['E'] == 4) &
        #                             # (calc_grps_df['col_var_id'] == 'erb') &
        #                             # (calc_grps_df['target_var_id'] == 'vieira') &
        #                             (calc_grps_df['weighted'] == False)
        #                             ].copy()
        arg_tuples = []
        for authors, auth_grp in calc_grps_df.groupby(['col_var_id', 'target_var_id']):
            print(f'authors: {authors}', file=sys.stdout, flush=True)
            auth_pair = f'{authors[0]}_{authors[1]}'
            calc_grps = auth_grp.to_dict(orient='records')

            calc_log2_df = calc_log2_df[~calc_log2_df['df_path'].str.contains('_pctiles')]


            for grp_d in calc_grps:
                # query_string = write_query_string(query_keys, grp_d)
                # query_d = {}
                # for key in query_keys:
                #     if isinstance(grp_d[key], str):
                #         query_d[key] = f'"{grp_d[key]}"'
                #     else:
                #         query_d[key] = grp_d[key]
                #
                # query_string = [f'{key}=={value}' for key, value in query_d.items()]
                # query_string = ' & '.join(query_string)
                grp_df = calc_log2_df.query(write_query_string(query_keys, grp_d))
                weighted_flag = get_weighted_flag(grp_d)
                grp_df = set_df_weighted(grp_df, weighted_flag)

                # if grp_d['weighted'] in ['True', True]:
                #     grp_df = grp_df[grp_df['weighted'] == True].copy()
                #     weighted_flag = 'weighted'
                # elif grp_d['weighted'] in ['False', False]:
                #     grp_df = grp_df[grp_df['weighted'] == False].copy()
                #     weighted_flag = 'unweighted'
                # print(f'grp_id:{str(grp_d["group_id"])} data loaded', file=sys.stdout, flush=True)

                # Filter for real data frames
                real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
                # print('cross_corr_sub', cross_corr_sub)
                # print('grp_df', real_dfs_references.head())
                arg_tuples.append((grp_d, config, calc_metric_dir, real_dfs_references,
                                   plot_flag, override, datetime_flag))

        num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([4, len(arg_tuples)])))
        print('num_cpus:', num_cpus, file=sys.stdout, flush=True)

        # Use multiprocessing to parallelize the process
        lag_dfs = []
        with Pool(num_cpus) as pool:
            results = pool.map(process_group, arg_tuples)
            # results = pool.map(process_group, arg_tuples)
            # pool.map(plot_slope_rhos_lib, arg_tuples)
        lag_df = pd.concat(results)
        lag_df = lag_df.reset_index(drop=True)

        csv_dir = calc_metric_dir / config.lag_processing.csvs
        csv_dir.mkdir(parents=True, exist_ok=True)

        csv_path = csv_dir / f'full_lag_maxes.csv'
        lag_df.to_csv(csv_path)


