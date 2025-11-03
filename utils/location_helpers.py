
from pathlib import Path
import os
import sys

def set_calc_path(args, proj_dir, config, second_suffix=''):
    calc_location = None
    if args is not None:
        if args.calc_dir is not None:
            calc_location = proj_dir/args.calc_dir

    if calc_location is None:
        if Path('/Users/jlanders').exists() == True:
            calc_location = proj_dir / (config.local.calc_carc + f'{second_suffix}')  #'calc_local_tmp'
        else:
            calc_location = proj_dir / (config.local.calc_carc + f'{second_suffix}')

    return calc_location

def set_output_path(args, calc_location, config):
    output_dir = None
    if args is not None:
        if args.output_dir is not None:
            output_dir = calc_location / args.output_dir

    if output_dir is None:
        try:
            # if config.carc.output_dir is not None:
            output_dir = calc_location / config.carc.output_dir
        except AttributeError:
            print('AttributeError: config.carc.output_dir not found', file=sys.stdout, flush=True)
            # output_dir = calc_location / 'calc_refactor'

        # if output_dir.exists() == False:
        #     output_dir = calc_location

    return output_dir

def replace(template, d):
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
    return template
import types
import collections.abc
import copy
def template_replace(template, d, return_replaced=True):
    replaced = []
    old_template = copy.copy(template)
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
        if template != old_template:
            replaced.append(key)
            old_template = copy.copy(template)
    if return_replaced is False:
        return template

    return template, replaced

def correct_iterable(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return [obj]
    else:
        if isinstance(obj, collections.abc.Iterable):
            return obj
        else:
            return [obj]

def check_csv_ext(output_file_name):
    if '.csv' not in output_file_name:
        output_file_name = f'{output_file_name}.csv'
    return output_file_name

def set_grp_path(parent_path, d, config=None, source='csv', grp_level='grp_dir_structure', make_grp=True):
    # print('source', source, file=sys.stdout, flush=True)
    if 'tp' in d.keys():
        d['Tp'] = d['tp']
    # print('tried to set grp path?')
    # print('set_grp_path', source)#, file=sys.stdout, flush=True)
    tmp_d = d.copy()
    tmp_d= {k:v[0] if isinstance(v, list) and len(v)==1 else v for k,v in tmp_d.items()}
    config_path = parent_path#set_model_config_path(parent_path, d, config=config)

    # print('set_grp_path d', d, source, file=sys.stdout, flush=True)
    if source == 'csv':
        if 'lag' in d and d['lag'] is not None:
            grp_level = 'dir_structure_csv'
        else:
            grp_level = 'grp_dir_structure'
        # grp_level = 'dir_structure_csv'
    else:
        grp_level = 'dir_structure'


    # print('grp_level', grp_level)#, file=sys.stdout, flush=True)

    if config is not None:
        try:
            grp_path_template = config.get_dynamic_attr("output.{var}", grp_level) # config.output.grp_dir_structure
            grp_path_template_filled = template_replace(grp_path_template,tmp_d, return_replaced=False)
            # grp_path_template_filled = '/'.join([path_part for path_part in grp_path_template_filled.split('/') if '{' not in path_part])# grp_path_template.replace('{col_var_id}', d["col_var_id"]).replace('{target_var_id}', d["target_var_id"]).replace('{E}', str(d["E"])).replace('{tau}', str(d["tau"])).replace('{knn}', 'knn'+str(d["knn"])).replace('{Tp}', str(d["Tp"])).replace('{lag}', str(d.get("lag",0)))
            grp_path = config_path / grp_path_template_filled
            # print('grp_path_template', grp_path_template, 'filled', grp_path_template_filled, '-> grp_path', grp_path, file=sys.stdout, flush=True)
            # if make_grp is True:
            #     grp_path.mkdir(exist_ok=True, parents=True)
        except:
            pass
    else:
        grp_path_template_filled = f'{d["col_var_id"]}_{d["target_var_id"]} / E{d["E"]}_tau{d["tau"]}'
        grp_path = config_path / grp_path_template_filled

    # print('set_grp_path d', grp_path, file=sys.stdout, flush=True)

    if make_grp is True:
        grp_path.mkdir(exist_ok=True, parents=True)

    return grp_path


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


def check_exists(file_name, calc_dir):
    dir_exists = os.path.exists(str(calc_dir))
    if dir_exists is False:
        return False, False

    calc_dir_list = os.listdir(str(calc_dir))#[entry.name for entry in calc_dir.iterdir() if entry.is_dir()]
    pset_id = file_name.split('_E')[0]  # assuming pset_id is the first part of the file name
    stem = file_name.split(pset_id)[1].lstrip('_')  # assuming the pset_id is before the first '__'
    pset_files = [fn for fn in calc_dir_list if fn.startswith(pset_id)]
    stem_files = [fn for fn in pset_files if fn.endswith(stem)]

    pset_exists = False
    stem_exists = False
    if str(file_name) in calc_dir_list:
        pset_exists = True
    if len(stem_files) > 0:
        stem_exists = True

    return pset_exists, stem_exists

def set_model_config_path(parent_path, d, config=None):
    # if "lag" not in d:
    #     d["lag"] = 0
    # if isinstance(d["lag"], str):
    #     if d["lag"].isdigit():
    #         d["lag"] = int(d["lag"])
    #     else:
    #         d["lag"] = 0

    if config is not None:
        try:
            dir_structure = config.output.non_grp_structure
            # print('dir_structure', dir_structure, file=sys.stdout, flush=True)
            dir_structure_filled = replace(dir_structure, d)
            # print('dir_structure_filled', parent_path / dir_structure_filled, file=sys.stdout, flush=True)
            return parent_path / dir_structure_filled
        except:
            pass
    return parent_path / f'knn_{d["knn"]}' / f'tp_{d["Tp"]}' / f'lag_{d["lag"]}'

def set_convergence_paths(calc_convergence_dir_parent, config, d):

    calc_convergence_dir = set_model_config_path(calc_convergence_dir_parent, d)
    calc_convergence_dir.mkdir(exist_ok=True, parents=True)

    calc_convergence_dir_csvs = calc_convergence_dir / config.calc_convergence_dir.dirs.csvs
    calc_convergence_dir_csvs.mkdir(exist_ok=True, parents=True)

    convergence_dir_csv_parts = calc_convergence_dir / f'{config.calc_convergence_dir.dirs.summary_frags}'
    convergence_dir_csv_parts.mkdir(exist_ok=True, parents=True)

    fig_dir = calc_convergence_dir / f'{config.calc_convergence_dir.dirs.figures}'
    fig_dir.mkdir(exist_ok=True, parents=True)

    return calc_convergence_dir, calc_convergence_dir_csvs, convergence_dir_csv_parts, fig_dir


def set_delta_rho_paths(config, calc_convergence_dir, second_suffix=''):

    delta_rho_dir = calc_convergence_dir / (config.calc_convergence_dir.dirs.delta_rho + second_suffix)
    delta_rho_dir.mkdir(exist_ok=True, parents=True)

    delta_rho_dir_csv_parts = delta_rho_dir / config.delta_rho_dir.dirs.summary_frags
    delta_rho_dir_csv_parts.mkdir(exist_ok=True, parents=True)

    return delta_rho_dir, delta_rho_dir_csv_parts


def set_raw_fig_dir(config, calc_convergence_dir, second_suffix=''):
    raw_fig_dir = calc_convergence_dir / (config.calc_carc_dir.dirs.ccm_surr_plots_dir_raw + second_suffix)
    raw_fig_dir.mkdir(exist_ok=True, parents=True)
    print('raw_fig_dir', raw_fig_dir, file=sys.stdout, flush=True)

    return raw_fig_dir