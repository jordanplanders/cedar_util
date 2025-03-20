# arg_parser.py

import argparse
from utils.data_processing import is_float
# from future.builtins import isinstance


def get_parser():
    """Return an argument parser with the common flags."""
    parser = argparse.ArgumentParser(description="Common parser for scripts")
    parser.add_argument('-v', '--verbose', action='store_true', help="Increase output verbosity")
    parser.add_argument('-j', '--project', required=True, type=str, help="Specify project name")
    parser.add_argument('-c', '--config', type=str, help="Specify config file name (no file type) (e.g. 'proj_config')")
    parser.add_argument('-p', '--parameters', type=str, help="parameter file name (no file type) (e.g. 'params_lag')")
    parser.add_argument('-t', '--test', action='store_true', help="If True, run in test mode")
    parser.add_argument('-o', '--override', dest='override', action='store_true', help="If True, rerun even if output file exist")
    parser.add_argument('-w', '--write', type=str, default='append', help="If append, add to existing, if replace, overwrite existing output file")
    parser.add_argument('-n', '--number', type=int, help="Specify a number")
    # parser.add_argument('-i', '--ind', type=int, help="Specify an index")
    parser.add_argument('-i', '--inds', nargs='+', type=int, help="List of integers")
    parser.add_argument('-r', '--target_rel_only', action='store_true',
                        help="Specify whether to only use target relation")
    parser.add_argument('-g', '--group_file',dest='group_file', type=str, help="Specify group csv file name")
    parser.add_argument('-a', '--vars', nargs='+',
                        type=str, help="List of strings. For make_params.py, expects col_var_id, target_var_id, or surrogates (any combination of:) col_var, target_var, neither")
    parser.add_argument('-f', '--file', dest='file', type=str, help="Specify file name")
    parser.add_argument('-l', '--flags',dest='flags',  nargs='+',
                        type=str, help="Specify any additional flags")
    parser.add_argument('-d', '--dir', dest='dir', type=str, help="Specify parent directory")


    return parser


def parse_flags(args, default_percent_threshold=.05, default_function_flag='binding',
                default_res_flag='', default_second_suffix=''):
    percent_threshold = None
    function_flag = None
    res_flag = None
    second_suffix = default_second_suffix

    if isinstance(args.flags, list):
        if 'inverse_exponential' in args.flags:
            function_flag = 'inverse_exponential'
        elif 'binding' in args.flags:
            function_flag = 'binding'

        for flag in args.flags:
            if 'coarse' in flag:
                res_flag = '_' + flag

        numeric_flags = [is_float(val) for val in args.flags if is_float(val) is not None]
        if len(numeric_flags) > 0:
            percent_threshold_candidates = [flag for flag in numeric_flags if flag <= 1]
            if len(percent_threshold_candidates) > 0:
                percent_threshold = percent_threshold_candidates[0]

            second_suffix_candidates = [flag for flag in numeric_flags if flag > 1]
            if len(second_suffix_candidates) > 0:
                second_suffix = f'_{int(second_suffix_candidates[0])}'

    if function_flag is None:
        function_flag = default_function_flag
    if percent_threshold is None:
        percent_threshold = default_percent_threshold
    if res_flag is None:
        res_flag = default_res_flag

    return percent_threshold, function_flag, res_flag, second_suffix


def construct_convergence_name(args, carc_config_d, percent_threshold, second_suffix):
    percent_threshold_label = str(percent_threshold * 100).lstrip('.0').replace('.', 'p')
    if '.' in percent_threshold_label:
        percent_threshold_label = '_' + percent_threshold_label.replace('.', 'p')

    calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}/tolerance{percent_threshold_label}'
    if isinstance(args.dir, str):
        if len(args.dir) > 0:
            subdir = args.dir
            calc_convergence_dir_name = f'{calc_convergence_dir_name}/{subdir}{second_suffix}'
    else:
        calc_convergence_dir_name = f'{calc_convergence_dir_name}{second_suffix}'

    return calc_convergence_dir_name