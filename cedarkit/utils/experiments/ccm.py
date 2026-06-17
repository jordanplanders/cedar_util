import os
import sys
import time

import pandas as pd
import pyEDM as pe
# from cedarkit.utils.cli.logging import print_log_line
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.workflow import process_output as po
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.workflow import process_output as po
    from utils.cli.logging import log_line



def run_experiment(arg_tuple):
    '''
    Run EDM CCM experiment based on CCMConfig object
    Parameters:
        arg_tuple: (ccm_obj, cpu_count, self_predict, time_offset)
            ccm_obj: CCMConfig object
            cpu_count: number of CPUs to use
            self_predict: boolean, whether to use self prediction
            time_offset: integer, offset to add to run_id
    Returns:
        (ccm_out_df, df_path)


    '''

    ccm_obj, script, start_ind  = arg_tuple

    time_var = ccm_obj.time_var
    run_id = int(time.time() * 1000) + start_ind

    df_path = ccm_obj.file_path#output_dir / df_csv_name
    log_line(logger, [f'{df_path} exists: {df_path.exists()}, starting start_ind {start_ind}',
                                     f'pset_id {ccm_obj.pset_id}, col_var_id {ccm_obj.col_var_id}',
                                     f'target_var_id {ccm_obj.target_var_id}, E {ccm_obj.E}, tau {ccm_obj.tau}',
                                     f'lag {ccm_obj.lag}, knn {ccm_obj.knn}, Tp {ccm_obj.Tp}, sample {ccm_obj.sample}',
                                     f'weighted {ccm_obj.weighted}, train_ind_i {ccm_obj.train_ind_i}',
                                     f'surr_var {ccm_obj.surr_var}, surr_num {ccm_obj.surr_num}'], indent=0, log_type="info")
    start_time = time.time()
    # start_time = print_log_line(script, 'run_experiment',
    #                             [f'{df_path} exists: {df_path.exists()}, starting start_ind {start_ind}',
    #                                  f'pset_id {ccm_obj.pset_id}, col_var_id {ccm_obj.col_var_id}',
    #                                  f'target_var_id {ccm_obj.target_var_id}, E {ccm_obj.E}, tau {ccm_obj.tau}',
    #                                  f'lag {ccm_obj.lag}, knn {ccm_obj.knn}, Tp {ccm_obj.Tp}, sample {ccm_obj.sample}',
    #                                  f'weighted {ccm_obj.weighted}, train_ind_i {ccm_obj.train_ind_i}',
    #                                  f'surr_var {ccm_obj.surr_var}, surr_num {ccm_obj.surr_num}'], 'info')

    try:
        pred_num = ccm_obj.pred_num
    except:
        pred_num = None

    # note: at some point "embedded=False" will not always be correct
    # cpu_count = 1 for HPC runs where resources are allocated less flexibly
    # changed to .tp from .Tp
    ccm_out = pe.CCM(dataFrame=ccm_obj.df,
                     E=ccm_obj.E, Tp=ccm_obj.tp, tau=-ccm_obj.tau,
                     exclusionRadius=ccm_obj.exclusion_radius,
                     knn=ccm_obj.knn, verbose=False,
                     columns=ccm_obj.col_var,
                     target=ccm_obj.target_var,
                     libSizes=ccm_obj.libsizes,
                     sample=ccm_obj.sample,
                     embedded=ccm_obj.embedded, seed=None,
                     weighted=ccm_obj.weighted, includeData=True, returnObject=True,
                     pred_num=pred_num,
                     num_threads=ccm_obj.cpus,
                     showPlot=False, noTime=ccm_obj.noTime, selfPredict=ccm_obj.self_predict)

    ccm_out_df = pd.concat(
        [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
    ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, ccm_obj.train_ind_i, ccm_obj.train_ind_f, lag=ccm_obj.lag)
    ccm_out_df['lag'] = ccm_obj.lag

    ccm_obj.output_path.mkdir(parents=True, exist_ok=True)

    ccm_out_df['run_id'] = run_id
    ccm_out_df['pset_id'] = ccm_obj.pset_id

    log_line(logger,[f'!\tfinish, start index: {start_ind}, {ccm_obj.pset_id}',
                                              f'time elapsed: {time.time() - start_time}',
                                              f'{ccm_obj.col_var_id}- {ccm_obj.target_var_id}; E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}'], indent=0, log_type="info")
          #    print_log_line(script, 'run_experiment', [f'!\tfinish, start index: {start_ind}, {ccm_obj.pset_id}',
          #                                     f'time elapsed: {time.time() - start_time}',
          #                                     f'{ccm_obj.col_var_id}- {ccm_obj.target_var_id}; E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}'],
          # 'info')

    # return [logs]
    return ccm_out_df, df_path


def write_to_file(ccm_out_df, df_path, overwrite=False):
    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len',
                   'train_ind_i']
    ccm_out_df = ccm_out_df[[col for col in ccm_out_df.columns if col not in remove_cols]].copy()
    df_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ccm_out_df_0 = pd.read_csv(df_path, index_col=0)
        if overwrite == False:
            ccm_out_df_0 = ccm_out_df_0[[col for col in ccm_out_df_0.columns if col not in remove_cols]].copy()
            ccm_out_df = pd.concat([ccm_out_df_0, ccm_out_df])
            ccm_out_df.reset_index(drop=True, inplace=True)
    except:
        pass

    ccm_out_df.to_csv(df_path)
    if os.path.exists(df_path):
        print('!\twrote to file: ', df_path)
    else:
        print('x\tfailed to write to file: ', df_path, file=sys.stderr, flush=True)
        print('x\tfailed to write to file: ', df_path, file=sys.stdout, flush=True)
