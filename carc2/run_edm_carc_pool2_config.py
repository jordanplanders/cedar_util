import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyEDM as pe
import yaml
from utils.arg_parser import get_parser
from utils.config_parser import load_config

from ccmtools.utils import process_output as po



def run_experiment(arg_tuple):
    pset, calc_dir, time_offset, start_ind, config = arg_tuple
    # with open(calc_dir.parent/'proj_config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # proj_d = config['proj_d']
    data_csv = config.raw_data.data_csv
    time_var = config.raw_data.time_var
    print('running', 'start_ind', start_ind, pset, file=sys.stdout, flush=True)
    pset['run_id'] = int(time.time() * 1000) + time_offset
    pset['pset_id'] = pset['id']
    # if 'col_var_id' not in pset:
    #     pset['col_var_id'] = pset[f'{config["col_var"]}_id']
    # if 'target_var_id' not in pset:
    #     pset['target_var_id'] = pset[f'{config["target_var"]}_id']
    run_config = SimpleNamespace(**pset)

    dir_path = calc_dir / str(run_config.pset_id)
    df_path = dir_path / 'df_{}.csv'.format(run_config.run_id)

    start_time = time.time()
    libSizes = np.arange(run_config.knn + 1, config.ccm_config.max_libsize, config.ccm_config.libsize_step)

    data_col_var = config.get_dynamic_attr("{var}.data_var", run_config.col_var_id)
    data_col_var_alias = config.get_dynamic_attr("{var}.var", run_config.col_var_id)

    data_target_var = config.get_dynamic_attr("{var}.data_var", run_config.target_var_id)
    data_target_var_alias = config.get_dynamic_attr("{var}.var", run_config.target_var_id)

    # Calculate exclusion_radius dynamically based on tau and E
    exclusion_radius = np.abs(run_config.tau) * run_config.E  #calculate_exclusion_radius(config.tau, config.E)
    run_config.exclusion_radius = exclusion_radius

    if run_config.Tp_flag != 'custom':
        if run_config.Tp_flag == 'excl_radius':
            run_config.Tp = exclusion_radius
        if run_config.Tp_flag == 'excl_radius_div2':
            run_config.Tp = int(exclusion_radius / 2)
        elif run_config.Tp_flag == 'tau':
            run_config.Tp = run_config.tau
    if run_config.Tp_flag == 'pairs':
        run_config['Tp'] = run_config.Tp_lag_total + run_config.lag

    # Load your data here
    data = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)
    data = data[[time_var, data_col_var, data_target_var]].rename(columns=
                                                                  {'time': 'date', data_col_var: data_col_var_alias,
                                                                   data_target_var: data_target_var_alias})
    _config_num_str = int(run_config.surr_num)

    surr_var_id = None
    if run_config.surr_var == config.col.var:
        surr_var_id = run_config.col_var_id
    if run_config.surr_var == config.target.var:  #TSI
        surr_var_id = run_config.target_var_id

    if surr_var_id is not None:
        surr_file = config.get_dynamic_attr("{var}.surr_file_name", surr_var_id)

        surr_data = pd.read_csv(proj_dir / 'surrogates' / surr_file, index_col=0)
        data = data.iloc[-len(surr_data):].copy()
        surr_var = config.get_dynamic_attr("{var}.var", surr_var_id)

        try:
            data[surr_var] = surr_data['{}_{}'.format(surr_var, _config_num_str)].values
        except:
            print('check lower v upper case of variable name in surr_file_name', surr_var, file=sys.stdout, flush=True)

    train_len = (len(data) - run_config.train_ind_i) - run_config.exclusion_radius

    # Load the data
    train_ind_f = min([run_config.train_ind_i + train_len + run_config.exclusion_radius, len(data)])
    df_sub = data.iloc[run_config.train_ind_i:train_ind_f].copy().reset_index()
    shifted = df_sub.copy()
    shifted[run_config.target_var] = shifted[run_config.target_var].shift(run_config.lag)
    shifted = shifted.dropna()

    try:
        pred_num = config.ccm_config.pred_num
    except:
        pred_num = None

    # note: at some point "embedded=False" will not always be correct
    # rever to run_config.sample
    ccm_out = pe.CCM(dataFrame=shifted,
                     E=run_config.E, Tp=run_config.Tp, tau=-run_config.tau, exclusionRadius=run_config.exclusion_radius,
                     knn=run_config.knn, verbose=False,
                     columns=run_config.col_var, target=run_config.target_var, libSizes=libSizes,
                     sample=run_config.sample,
                     embedded=False, seed=None,
                     weighted=run_config.weighted, includeData=True, returnObject=True,
                     pred_num=pred_num,
                     # num_threads=6,
                     showPlot=False, noTime=False)

    ccm_out_df = pd.concat(
        [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
    ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, run_config.train_ind_i, train_ind_f, lag=run_config.lag)
    ccm_out_df['lag'] = run_config.lag

    dir_path.mkdir(parents=True, exist_ok=True)

    ccm_out_df['run_id'] = run_config.run_id
    ccm_out_df['pset_id'] = run_config.pset_id

    ccm_out_df.to_csv(df_path)
    print('finish', 'start index:', start_ind, run_config.pset_id, f'time elapsed: {time.time() - start_time}',
          run_config.col_var_id, run_config.target_var_id,
          'E, tau, lag= ', run_config.E, run_config.tau, run_config.lag,
          file=sys.stdout, flush=True)

    # return [logs]


def check_exists(pset_id, calc_dir):
    calc_dir_list = [entry.name for entry in calc_dir.iterdir() if entry.is_dir()]
    if str(pset_id) in calc_dir_list:
        return True
    else:
        return False


if __name__ == '__main__':
    # grab parameter file
    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    gen_config = 'proj_config'
    config = load_config(proj_dir / f'{gen_config}.yaml')

    # print("Script2 in verbose mode")
    if args.parameters is not None:
        parameter_flag = args.parameters
    else:
        print('parameters are required', file=sys.stdout, flush=True)
        print('parameters are required', file=sys.stderr, flush=True)
        sys.exit(0)

    parameter_dir = proj_dir / 'parameters'
    parameter_path = parameter_dir / f'{parameter_flag}.csv'
    parameter_df = pd.read_csv(parameter_path)

    # # alert!!!!
    flags = []
    if args.flags is not None:
        flags += args.flags

    num_inds = 10
    if args.inds is not None:
        start_ind = int(args.inds[-1])
        if len(args.inds) > 1:
            num_inds = int(args.inds[-2])
    end_ind = start_ind + num_inds

    print(parameter_df.index, file=sys.stdout, flush=True)
    if start_ind not in parameter_df.index:  #len(parameter_ds):
        print('start_ind is not in available indexes', file=sys.stdout, flush=True)
        sys.exit(0)

    spec_config = {}
    if args.config is not None:
        spec_config_yml = args.config
        config = load_config(proj_dir / f'{spec_config_yml}.yaml')

    if args.override:
        override = args.override
    else:
        override = False

    parameter_df = parameter_df.loc[start_ind:end_ind, :].copy()
    # parameter_df = parameter_df[
    #     # (parameter_df['surr_var']=='neither')&
    #     (parameter_df['surr_num'].isin([1, 2]))].copy()
    parameter_ds = parameter_df.to_dict(orient='records')

    second_suffix = ''
    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}'

    calc_dir = proj_dir / (config.calc_dir + f'{second_suffix}')
    calc_dir.mkdir(parents=True, exist_ok=True)

    calc_dir_list = [entry.name for entry in calc_dir.iterdir() if entry.is_dir()]

    logs = []
    log_element = []
    if Path('/Users/jlanders').exists():
        print('local', file=sys.stdout, flush=True)

        arg_tuples = []
        for time_offset, pset in enumerate(parameter_ds):
            # if 'col_var_id' not in pset:
            #     pset['col_var_id'] = pset[f'{str.lower(config[col_var"])}_id']
            # if 'target_var_id' not in pset:
            #     pset['target_var_id'] = pset[f'{str.lower(config["target_var"])}_id']

            if override == True:
                arg_tuples.append((pset, calc_dir, time_offset, start_ind + time_offset, config))
            else:
                if 'add' in flags:
                    arg_tuples.append((pset, calc_dir, time_offset, start_ind + time_offset, config))
                if str(pset['id']) in calc_dir_list:
                    print('skipping', pset['id'], pset['col_var_id'], pset['target_var_id'], pset['E'], pset['tau'],
                          pset['lag'], file=sys.stdout, flush=True)
                else:
                    # if check_exists(pset['id'], calc_dir) == False:
                    arg_tuples.append((pset, calc_dir, time_offset, start_ind + time_offset, config))

        for arg in arg_tuples:
            run_experiment(arg)

    else:
        parameter_ds = parameter_ds[0]
        time_offset = 1

        pset = parameter_ds
        if 'col_var_id' not in pset:
            pset['col_var_id'] = pset[f'{config["col_var"]}_id']
        if 'target_var_id' not in pset:
            pset['target_var_id'] = pset[f'{config["target_var"]}_id']
        arg_tuples = []
        # for time_offset, pset in enumerate(parameter_ds):
        if override == False:
            if str(pset['id']) in calc_dir_list:
                print('skipping', pset['id'], pset['col_var_id'], pset['target_var_id'], pset['E'], pset['tau'],
                      pset['lag'], file=sys.stderr, flush=True)
            else:
                print('prepping', pset['id'], pset['col_var_id'], pset['target_var_id'], pset['E'], pset['tau'],
                      pset['lag'], file=sys.stdout, flush=True)
                arg_tuples.append((pset, calc_dir, time_offset, start_ind, config))
        else:
            arg_tuples.append((pset, calc_dir, time_offset, start_ind, config))
            print('prepping', pset['id'], pset['col_var_id'], pset['target_var_id'], pset['E'], pset['tau'],
                  pset['lag'], file=sys.stdout, flush=True)

        for arg in arg_tuples:
            print('sending', arg[0], arg[-1], file=sys.stdout, flush=True)
            run_experiment(arg)
