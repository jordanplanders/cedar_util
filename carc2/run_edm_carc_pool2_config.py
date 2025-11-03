import time
from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import pyEDM as pe
from datetime import datetime

from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.location_helpers import *
from utils.data_access import pull_raw_data
from utils.location_helpers import check_exists
from utils.run_tools import decide_file_handling

from ccm_utils import process_output as po

def get_run_config(pset):
    run_config = SimpleNamespace(**pset)
    return run_config


def print_log_line(script, function, log_line, log_type='info'):
    if log_type == 'error':
        file_pointer = sys.stderr
    else:
        file_pointer = sys.stdout
    timestamp = datetime.now()
    print(timestamp.strftime('%Y-%m-%d %H:%M:%S'), log_line, f'{script}: {function}', file=file_pointer, flush=True)
    return time.time()


# def template_replace(template, d, return_replaced=True):
#     replaced = []
#     old_template = copy.copy(template)
#     for key, value in d.items():
#         template = template.replace(f'{{{key}}}', str(value))
#         if template != old_template:
#             replaced.append(key)
#             old_template = copy.copy(template)
#     if return_replaced is False:
#         return template
#
#     return template, replaced

def run_experiment(arg_tuple):
    pset, output_dir, output_file_name, time_offset, start_ind, config, proj_dir, cpu_count, self_predict = arg_tuple


    time_var = config.raw_data.time_var
    pset['run_id'] = int(time.time() * 1000) + time_offset
    if 'id' in pset:
        pset['pset_id'] = pset['id']

    run_config = get_run_config(pset)

    df_csv_name = check_csv_ext(output_file_name)
    df_path = output_dir / df_csv_name
    start_time = print_log_line('run_edm_carc_pool2_config', 'run_experiment',
                                     f'{df_path} exists: {df_path.exists()}, starting start_ind {start_ind}, pset_id {run_config.pset_id}, col_var_id {run_config.col_var_id}, target_var_id {run_config.target_var_id}, E {run_config.E}, tau {run_config.tau}, lag {run_config.lag}, knn {run_config.knn}, Tp {run_config.Tp}, sample {run_config.sample}, weighted {run_config.weighted}, train_ind_i {run_config.train_ind_i}, surr_var {run_config.surr_var}, surr_num {run_config.surr_num}', 'info')

    # start_time = time.time()
    libSizes = np.arange(run_config.knn + 1, config.ccm_config.max_libsize, config.ccm_config.libsize_step)

    # Calculate exclusion_radius dynamically based on tau and E
    exclusion_radius = np.abs(run_config.tau) * run_config.E  #calculate_exclusion_radius(config.tau, config.E)
    run_config.exclusion_radius = exclusion_radius

    # if run_config.Tp_flag != 'custom':
    #     if run_config.Tp_flag == 'excl_radius':
    #         run_config.Tp = exclusion_radius
    #     if run_config.Tp_flag == 'excl_radius_div2':
    #         run_config.Tp = int(exclusion_radius / 2)
    #     elif run_config.Tp_flag == 'tau':
    #         run_config.Tp = run_config.tau
    # if run_config.Tp_flag == 'pairs':
    #     run_config['Tp'] = run_config.Tp_lag_total + run_config.lag

    # Load your data here
    data = pull_raw_data(config, proj_dir,[ run_config.col_var_id, run_config.target_var_id])
    # data = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)
    # data = data[[time_var, data_col_var, data_target_var]].rename(columns=
    #                                                               {'time': 'date', data_col_var: data_col_var_alias,
    #                                                                data_target_var: data_target_var_alias})
    _config_num_str = int(run_config.surr_num)

    surr_var_id = None
    if run_config.surr_var == config.col.var:
        surr_var_id = run_config.col_var_id
    if run_config.surr_var == config.target.var:  #TSI
        surr_var_id = run_config.target_var_id

    if surr_var_id is not None:
        surr_file = config.get_dynamic_attr("{var}.surr_file_name", surr_var_id)
        print('surr', surr_file, file=sys.stdout, flush=True)
        surr_file = surr_file[0] if isinstance(surr_file, list) else surr_file

        surr_data = pd.read_csv(proj_dir / 'surrogates' / surr_file, index_col=0)
        surr_data = surr_data.rename(columns={time_var: 'date' })
        data = data.rename(columns={time_var: 'date' })

        # data = data.iloc[-len(surr_data):].copy()
        surr_var = config.get_dynamic_attr("{var}.var", surr_var_id)
        surr_col_name = '{}_{}'.format(surr_var, _config_num_str)

        surr_data = surr_data[['date', surr_col_name]].copy()
        surr_data = pd.merge(surr_data, data, on='date', how='outer')
        data_cols = [col for col in data.columns if col not in ['date']] + [surr_col_name]
        surr_data = surr_data.dropna(subset=data_cols, how='all')
        surr_data = surr_data.drop(columns=[surr_var], axis=1)
        surr_data = surr_data.rename(columns={surr_col_name: surr_var})
        if surr_var not in surr_data.columns:
            print('\tsurr_var not in surr_data', surr_var,'. Check for case sensitivity in surr_file_name',  file=sys.stderr, flush=True)
        else:
            data= surr_data.copy()

    train_len = (len(data) - run_config.train_ind_i) - run_config.exclusion_radius

    # Load the data
    shifted = data.copy()
    lag = run_config.lag if np.isnan(run_config.lag)==False else 0
    shifted[run_config.target_var] = shifted[run_config.target_var].shift(lag)
    shifted = shifted.dropna()
    train_ind_f = min([run_config.train_ind_i + train_len + run_config.exclusion_radius, len(shifted)])
    shifted = shifted.iloc[run_config.train_ind_i:train_ind_f].copy().reset_index()
    #
    #
    # train_ind_f = min([run_config.train_ind_i + train_len + run_config.exclusion_radius, len(data)])
    # df_sub = data.iloc[run_config.train_ind_i:train_ind_f].copy().reset_index()
    # shifted = df_sub.copy()
    # lag = run_config.lag if np.isnan(run_config.lag)==False else 0
    # shifted[run_config.target_var] = shifted[run_config.target_var].shift(lag)
    # shifted = shifted.dropna()

    try:
        pred_num = config.ccm_config.pred_num
    except:
        pred_num = None

    # note: at some point "embedded=False" will not always be correct
    # rever to run_config.sample
    # cpu_allocation = os.cpu_count() if os.cpu_count() is not None and os.cpu_count()<17  else 16
    # print('cpu_count', cpu_count, file=sys.stdout, flush=True)
    ccm_out = pe.CCM(dataFrame=shifted,
                     E=run_config.E, Tp=run_config.Tp, tau=-run_config.tau, exclusionRadius=run_config.exclusion_radius,
                     knn=run_config.knn, verbose=False,
                     columns=run_config.col_var, target=run_config.target_var, libSizes=libSizes,
                     sample=run_config.sample,
                     embedded=False, seed=None,
                     weighted=run_config.weighted, includeData=True, returnObject=True,
                     pred_num=pred_num,
                     num_threads=cpu_count,
                     showPlot=False, noTime=False, selfPredict=self_predict)

    ccm_out_df = pd.concat(
        [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
    ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, run_config.train_ind_i, train_ind_f, lag=run_config.lag)
    ccm_out_df['lag'] = run_config.lag

    output_dir.mkdir(parents=True, exist_ok=True)

    ccm_out_df['run_id'] = run_config.run_id
    ccm_out_df['pset_id'] = run_config.pset_id

    # ccm_out_df.to_csv(df_path)
    print('!\tfinish', 'start index:', start_ind, run_config.pset_id, f'time elapsed: {time.time() - start_time}',
          run_config.col_var_id, run_config.target_var_id,
          'E, tau, lag= ', run_config.E, run_config.tau, run_config.lag,
          file=sys.stdout, flush=True)

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
        parameter_dir = proj_dir / 'parameters'
        parameter_path = parameter_dir / f'{parameter_flag}.csv'
        parameter_df = pd.read_csv(parameter_path)
    else:
        print('parameters are required', file=sys.stdout, flush=True)
        print('parameters are required', file=sys.stderr, flush=True)
        sys.exit(0)


    # # alert!!!!
    flags = []
    if args.flags is not None:
        flags += args.flags

    if args.cpus is not None:
        cpu_count = int(args.cpus)
    else:
        cpu_count = 4

    num_inds = 10
    if args.inds is not None:
        start_ind = int(args.inds[-1])
        if len(args.inds) > 1:
            num_inds = int(args.inds[-2])
    end_ind = start_ind + num_inds

    if start_ind not in parameter_df.index:  #len(parameter_ds):
        print('start_ind is not in available indexes', file=sys.stdout, flush=True)
        sys.exit(0)

    spec_config = {}
    if args.config is not None:
        spec_config_yml = args.config
        config = load_config(proj_dir / f'{spec_config_yml}.yaml')

    parameter_df = parameter_df.loc[start_ind:end_ind, :].copy()
    for int_col in ['E', 'tau', 'lag', 'knn', 'train_ind_i', 'Tp', 'Tp_lag_total', 'sample', 'id']:
        if int_col in parameter_df.columns:
            parameter_df[int_col] = parameter_df[int_col].astype(int)
    parameter_ds = parameter_df.to_dict(orient='records')

    second_suffix = ''
    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}'

    calc_location = set_calc_path(args, proj_dir, config, second_suffix)
    output_dir = set_output_path(args, calc_location, config)
    calc_location.mkdir(parents=True, exist_ok=True)

    logs = []
    log_element = []
    self_predict = False
    if Path('/Users/jlanders').exists():
        print('local', file=sys.stdout, flush=True)

        arg_tuples = []
        for time_offset, pset_d in enumerate(parameter_ds):
            print(pset_d, file=sys.stdout, flush=True)
            pset_d= {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in pset_d.items()}
            if 'pset_id' not in pset_d:
                if 'id' in pset_d:
                    pset_d['pset_id'] = pset_d['id']

            print('before calc sub', pset_d, file=sys.stdout, flush=True)
            calc_sub_location = set_grp_path(output_dir, pset_d, config=config)# /f'tp_{pset_d["Tp"]}'/f'{pset_d["col_var_id"]}_{pset_d["target_var_id"]}' / f'E{pset_d["E"]}_tau{pset_d["tau"]}'
            # calc_sub_location.mkdir(parents=True, exist_ok=True)
            # calc_dir_list = os.listdir(calc_sub_location)

            print('before try', pset_d, file=sys.stdout, flush=True)
            try:
                print('inside try', pset_d, file=sys.stdout, flush=True)
                file_name_template = config.output.file_format_csv
                file_name = template_replace(file_name_template, pset_d, return_replaced=False)# f'{replace(file_name_template, pset_d)}.csv'
                print('file_name from template', file_name,pset_d, file=sys.stdout, flush=True)
            except:
                file_name = f"{pset_d['pset_id']}_E{pset_d['E']}_tau{pset_d['tau']}__{pset_d['surr_var']}{pset_d['surr_num']}.csv"

            pset_exists, stem_exists = check_exists(file_name, calc_sub_location) # if file_name in calc_dir_list else (False, False)
            existence = pset_exists # this is strong existence criteria... if want to check for stem existence, use stem_exists

            run_continue, overwrite = decide_file_handling(args, existence)
            # print('run_continue', run_continue, 'overwrite', overwrite,file_name,  file=sys.stdout, flush=True)
            # print('calc_dir_list', calc_dir_list, file=sys.stdout, flush=True)

            if run_continue == False:
                print(file_name, '\n\tskipping: ', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], f'E={pset_d["E"]}, tau={pset_d["tau"]}',
                      pset_d['lag'], file=sys.stdout, flush=True)
                continue

            # print(file_name, calc_sub_location, calc_dir_list, '\nrun_continue: ', run_continue, ', overwrite: ', overwrite,f'-- {file_name}', '\n\trunning ', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'],
            #       pset_d['tau'],
            #       pset_d['lag'], file=sys.stdout, flush=True)
            print('how many cores:', os.cpu_count(), file=sys.stdout, flush=True)
            candidate_tuple = (pset_d, calc_sub_location, file_name, time_offset, start_ind + time_offset, config, proj_dir, cpu_count, self_predict)
            write_to_file(*run_experiment(candidate_tuple), overwrite=overwrite)


            # if override == True:
            #     arg_tuples.append(candidate_tuple)
            # else:
            #     if 'add' in flags:
            #         arg_tuples.append(candidate_tuple)
            #
            #     file_name = f"{pset_d['pset_id']}_E{pset_d['E']}_tau{pset_d['tau']}__{pset_d['surr_var']}{pset_d['surr_num']}.csv"
            #     if str(file_name) in calc_dir_list:
            #         print('skipping', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'], pset_d['tau'],
            #               pset_d['lag'], file=sys.stdout, flush=True)
            #     else:
            #         # if check_exists(pset['id'], calc_dir) == False:
            #         arg_tuples.append(candidate_tuple)

        # for arg in arg_tuples:
        #     write_to_file(*run_experiment(arg))

    else:
        parameter_ds = parameter_ds[0]
        time_offset = 1

        pset_d = parameter_ds
        if 'pset_id' not in pset_d:
            if 'id' in pset_d:
                pset_d['pset_id'] = pset_d['id']
        if 'col_var_id' not in pset_d:
            pset_d['col_var_id'] = pset_d[f'{config["col_var"]}_id']
        if 'target_var_id' not in pset_d:
            pset_d['target_var_id'] = pset_d[f'{config["target_var"]}_id']

        arg_tuples = []
        # for time_offset, pset in enumerate(parameter_ds):
        calc_sub_location = set_grp_path(output_dir, pset_d, config=config) #calc_location / f'{pset_d["col_var_id"]}_{pset_d["target_var_id"]}' / f'E{pset_d["E"]}_tau{pset_d["tau"]}'
        # calc_sub_location.mkdir(parents=True, exist_ok=True)
        # calc_dir_list = os.listdir(calc_sub_location)

        try:
            file_name_template = config.output.file_format_csv
            file_name = template_replace(file_name_template, pset_d, return_replaced=False)# f'{replace(file_name_template, pset_d)}.csv'
        except:
            file_name = f"{pset_d['pset_id']}_E{pset_d['E']}_tau{pset_d['tau']}__{pset_d['surr_var']}{pset_d['surr_num']}.csv"

        pset_exists, stem_exists = check_exists(file_name,
                                                calc_sub_location)  # if file_name in calc_dir_list else (False, False)
        existence = pset_exists  # this is strong existence criteria... if want to check for stem existence, use stem_exists
        print('file_name', file_name, 'existence', existence, file=sys.stdout, flush=True)
        run_continue, overwrite = decide_file_handling(args, existence)

        if run_continue == False:
            print('skipping', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'], pset_d['tau'],
                  pset_d['lag'], file=sys.stdout, flush=True)
        else:
            candidate_tuple = (pset_d, calc_sub_location, file_name, time_offset, start_ind + time_offset, config, proj_dir, cpu_count, self_predict)
            print('prepping', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'],
                  pset_d['tau'],
                  pset_d['lag'], file=sys.stdout, flush=True)

            print('sending', candidate_tuple[0], candidate_tuple[-1], file=sys.stdout, flush=True)
            write_to_file(*run_experiment(candidate_tuple), overwrite=overwrite)
            # arg_tuples.append(candidate_tuple)

        # if override == False:
        #     if existence_continue == False:
        #     # if file_name in calc_dir_list:
        #         print('\talready written', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'], pset_d['tau'],
        #               pset_d['lag'], file=sys.stderr, flush=True)
        #     else:
        #         print('prepping', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'], pset_d['tau'],
        #               pset_d['lag'], file=sys.stdout, flush=True)
        #         arg_tuples.append(candidate_tuple)
        # else:
        #     arg_tuples.append((pset_d, calc_sub_location, time_offset, start_ind, config, proj_dir))
        #     print('prepping', pset_d['pset_id'], pset_d['col_var_id'], pset_d['target_var_id'], pset_d['E'], pset_d['tau'],
        #           pset_d['lag'], file=sys.stdout, flush=True)

        # for arg in arg_tuples:
        #     print('sending', arg[0], arg[-1], file=sys.stdout, flush=True)
        #     write_to_file(*run_experiment(arg))
