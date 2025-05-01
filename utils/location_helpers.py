
from pathlib import Path
import os
import sys

def set_calc_path(args, proj_dir, config, second_suffix=''):
    if args.calc_dir is not None:
        calc_location = proj_dir/args.calc_dir
    else:
        if Path('/Users/jlanders').exists() == True:
            calc_location = proj_dir / (config.local.calc_carc + f'{second_suffix}')  #'calc_local_tmp'
        else:
            calc_location = proj_dir / (config.carc.calc_carc + f'{second_suffix}')

    return calc_location

def set_output_path(args, calc_location, config):
    if args.output_dir is not None:
        output_dir = calc_location / args.output_dir
    elif config.carc.output_dir is not None:
        output_dir = calc_location / config.carc.output_dir
    else:
        output_dir = calc_location
    return output_dir

def set_proj_dir(proj_name, current_path):
    if proj_name in str(current_path):
        proj_dir = Path(str(current_path).split(proj_name)[0]) / proj_name
    else:
        proj_dir = current_path / proj_name

    if 'proj_config.yaml' in os.listdir(proj_dir):
        return proj_dir
    else:
        print('proj_config.yaml not found in project directory', file=sys.stdout, flush=True)
        print('proj_config.yaml not found in project directory', file=sys.stderr, flush=True)
        sys.exit(0)


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
