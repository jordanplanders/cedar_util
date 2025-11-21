import time
import sys
import os
import pandas as pd
from pathlib import Path

try:
    from cedarkit.core.project_config import load_config
    from cedarkit.utils.cli.arg_parser import get_parser
    from cedarkit.utils.experiments.ccm import run_experiment, write_to_file
    from cedarkit.core.data_objects import CCMConfig
    from cedarkit.utils.routing.paths import set_calc_path, set_output_path, check_location
    from cedarkit.utils.io.gonogo import decide_file_handling

except ImportError:
    # Fallback: imports when running as a package
    from core.project_config import load_config
    from utils.cli.arg_parser import get_parser
    from utils.experiments.ccm import run_experiment, write_to_file
    from utils.routing.paths import set_calc_path, set_output_path, check_location
    from core.data_objects import CCMConfig
    from utils.io.gonogo import decide_file_handling

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

    if args.parameters is not None:
        parameter_flag = args.parameters
        parameter_dir = proj_dir / 'parameters'
        parameter_path = parameter_dir / f'{parameter_flag}.csv'
        parameter_df = pd.read_csv(parameter_path)
    else:
        print('parameters are required', file=sys.stdout, flush=True)
        print('parameters are required', file=sys.stderr, flush=True)
        sys.exit(0)

    flags = []
    if args.flags is not None:
        flags += args.flags

    if args.cpus is not None:
        cpu_count = int(args.cpus)
    else:
        cpu_count = 4

    #
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

    loc = check_location(local_word='jlanders')
    # if Path('/Users/jlanders').exists():
    if loc == 'local':
        print('local', file=sys.stdout, flush=True)

        arg_tuples = []
        for time_offset, pset_d in enumerate(parameter_ds):
            print(pset_d, file=sys.stdout, flush=True)
            pset_d= {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in pset_d.items()}
            if 'pset_id' not in pset_d:
                if 'id' in pset_d:
                    pset_d['pset_id'] = pset_d['id']

            # new_rc = RunConfig(pset_d)
            # ccm_obj = CCMConfig(new_rc, config, proj_dir=proj_dir)
            ccm_obj = CCMConfig(pset_d, config, proj_dir=proj_dir)

            pset_exists, stem_exists = ccm_obj.check_run_exists()
            # this is strong existence criteria... if want to check for stem existence, use stem_exists

            run_continue, overwrite = decide_file_handling(args, pset_exists)

            if run_continue == False:
                print(ccm_obj.file_name, '\n\tskipping: ', ccm_obj.pset_id, ccm_obj.col_var_id, ccm_obj.target_var_id, f'E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}', file=sys.stdout, flush=True)
                continue

            print('how many cores:', cpu_count, file=sys.stdout, flush=True)
            ccm_obj.cpus =cpu_count
            ccm_obj.self_predict = self_predict

            ccm_out_df, df_path = ccm_obj.run_ccm(overwrite=overwrite, ind=start_ind + time_offset)
            # ccm_obj.id_num = start_ind + time_offset
            # candidate_tuple = (ccm_obj, cpu_count, self_predict, start_ind + time_offset)#(ccm_obj, time_offset, start_ind + time_offset, config, cpu_count, self_predict)
            # write_to_file(*run_experiment(ccm_obj, ind = start_ind + time_offset), overwrite=overwrite)

    else:
        parameter_ds = parameter_ds[0]
        time_offset = 1

        pset_d = parameter_ds

        ccm_obj = CCMConfig(pset_d, config, proj_dir=proj_dir)
        pset_exists, stem_exists = ccm_obj.check_run_exists()
        # this is strong existence criteria... if want to check for stem existence, use stem_exists

        run_continue, overwrite = decide_file_handling(args, pset_exists)

        if run_continue == False:
            print(ccm_obj.file_name, '\n\tskipping: ', ccm_obj.pset_id, ccm_obj.col_var_id, ccm_obj.target_var_id,
                  f'E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}', file=sys.stdout, flush=True)
        else:
            print('how many cores:', cpu_count, file=sys.stdout, flush=True)
            # candidate_tuple = (ccm_obj, cpu_count, self_predict,
            #                    start_ind + time_offset)  # (ccm_obj, time_offset, start_ind + time_offset, config, cpu_count, self_predict)
            # write_to_file(*run_experiment(candidate_tuple), overwrite=overwrite)
            ccm_obj.cpus =cpu_count
            ccm_obj.self_predict = self_predict

            ccm_out_df, df_path = ccm_obj.run_ccm(overwrite=overwrite, ind=start_ind + time_offset)
            # ccm_obj.id_num = start_ind + time_offset
            # candidate_tuple = (ccm_obj, cpu_count, self_predict, start_ind + time_offset)#(ccm_obj, time_offset, start_ind + time_offset, config, cpu_count, self_predict)
            # write_to_file(*run_experiment(ccm_obj, ind=start_ind + time_offset), overwrite=overwrite)
