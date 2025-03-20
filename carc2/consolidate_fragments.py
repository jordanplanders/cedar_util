
import pandas as pd
from pathlib import Path
import os
import sys

import time

from utils.arg_parser import get_parser, parse_flags, construct_convergence_name
from utils.config_parser import load_config
from utils.data_access import collect_raw_data, get_weighted_flag, set_df_weighted, write_query_string

from utils.data_processing import is_float

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

    override = args.override if args.override is not None else False
    #     override = args.override
    # else:
    #     override = False

    # second_suffix = ''
    # if args.test:
    second_suffix = f'_{int(time.time() * 1000)}' if args.test is True else ''

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    percent_threshold, function_flag, res_flag, second_suffix = parse_flags(args,
                                                                            default_percent_threshold=.05,
                                                                            default_function_flag='binding',
                                                                            default_res_flag='',
                                                                            default_second_suffix=second_suffix)

    calc_convergence_dir_name = construct_convergence_name(args, carc_config_d, percent_threshold, second_suffix)
    calc_convergence_dir = calc_location / calc_convergence_dir_name
    # calc_convergence_dir_name = carc_config_d.dirs.calc_convergence_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    consolidate_targets = []
    # res_flag = ''
    # percent_threshold = None

    # function_flag=None
    # if isinstance(args.flags, list):
    #     if 'binding' in args.flags:
    #         function_flag = 'binding'
    #
    #     for flag in args.flags:
    #         if 'coarse' in flag:
    #             res_flag= '_'+flag
    #
    #     numeric_flags= [is_float(val) for val in args.flags if is_float(val) is not None]
    #     if len(numeric_flags)>0:
    #         # move all percent threshold to args.dir
    #         # percent_threshold_candidates = [flag for flag in numeric_flags if flag<=1]
    #         # if len(percent_threshold_candidates) > 0:
    #         #     percent_threshold = percent_threshold_candidates[0]
    #         #     percent_threshold_label = str(percent_threshold * 100).lstrip('.0').replace('.', 'p')
    #         #     if '.' in percent_threshold_label:
    #         #         percent_threshold_label = '_' + percent_threshold_label.replace('.', 'p')
    #
    #         second_suffix_candidates = [flag for flag in numeric_flags if flag>1]
    #         if len(second_suffix_candidates) > 0:
    #             second_suffix = f'_{int(second_suffix_candidates[0])}'

    # if percent_threshold is not None:


    # calc_convergence_dir_name = carc_config_d.dirs.calc_convergence_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'
    # if function_flag is not None:
    #     subdir = f'{function_flag}{res_flag}'
    #     calc_convergence_dir = calc_location / calc_convergence_dir_name / f'{function_flag}{res_flag}{second_suffix}'
    # else:
    #     calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}{second_suffix}'
    #     calc_convergence_dir = calc_location / calc_convergence_dir_name

    # calc_convergence_dir_csvs = calc_convergence_dir / config.calc_convergence_dir.dirs.csvs
    #
    # if percent_threshold is not None:
    #     perc_threshold_dir = calc_convergence_dir / f'{percent_threshold_label}'
    # else:
    #     perc_threshold_dir = calc_convergence_dir

    # calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}{second_suffix}'
    # calc_convergence_dir = calc_location / calc_convergence_dir_name
    # if isinstance(args.dir, str):
    #     if len(args.dir) > 0:
    #         subdir = args.dir
    #         calc_convergence_dir = calc_location / carc_config_d.dirs.calc_convergence_dir / f'{subdir}{second_suffix}'
    #         # perc_threshold_dir = perc_threshold_dir / args.dir

    print(calc_convergence_dir)
    convergence_dir_csv_parts = calc_convergence_dir / config.calc_convergence_dir.dirs.summary_frags
    convergence_csv = calc_convergence_dir / f'{config.calc_convergence_dir.csvs.convergence_metrics_csv}.csv'
    print(convergence_csv, file=sys.stdout, flush=True)
    delta_rho_dir = calc_convergence_dir/config.calc_convergence_dir.dirs.delta_rho
    delta_rho_dir_csv_parts = delta_rho_dir/config.delta_rho_dir.dirs.summary_frags
    delta_rho_csv_name = f'{config.calc_convergence_dir.csvs.delta_rho_csv}.csv'
    delta_rho_csv = calc_convergence_dir / delta_rho_csv_name

    if isinstance(args.flags, list):
        if 'deltarho' in args.flags:
            if (delta_rho_dir_csv_parts).exists() == True:
                fragments = [pd.read_csv(delta_rho_dir_csv_parts/frag_csv) for frag_csv in os.listdir(delta_rho_dir_csv_parts)]
                frag_df = pd.concat(fragments).reset_index(drop=True)
                drop_cols = [col for col in frag_df.columns if 'Unnamed' in col]
                frag_df.drop(columns=drop_cols, inplace=True)
                if 'Unnamed: 0' in frag_df.columns:
                    frag_df.drop(columns=['Unnamed: 0'], inplace=True)
                frag_df.to_csv(delta_rho_csv)

        if 'convergence' in args.flags:
            if (convergence_dir_csv_parts).exists() == True:
                fragments = []
                for frag_csv in os.listdir(convergence_dir_csv_parts):
                    frag = pd.read_csv(convergence_dir_csv_parts / frag_csv, index_col=0)
                    frags = frag.to_dict(orient='records')

                    # deprecated
                    # if 'approach2' not in args.flags:
                    #     if len(frag) > 1:
                    #         row = frags[-1]
                    #     else:
                    #         row = frags[0]
                    #
                    #     row['min_lib_size']=20
                    #     if 'max_lib_size' not in row.keys():
                    #         row['max_lib_size'] = 350
                    # else:
                    row = [frag for frag in frags if frag['curve_fit'] == 'middle_envelope'][-1]
                    fragments.append(row)
                    frag_df = pd.DataFrame(fragments)#.reset_index(drop=True)
                    frag_df.to_csv(convergence_csv)