# arg_parser.py

import argparse


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
