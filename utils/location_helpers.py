
from pathlib import Path
import os
import sys
import types
import collections.abc
import copy


def check_location(target_path=None, local_word='jlanders'):
    if target_path is None:
        target_path = Path.cwd()

    if local_word in str(target_path):
        return 'local'
    else:
        return 'hpc'


def set_calc_path(args, proj_dir, config, second_suffix=''):
    calc_location = None
    if args is not None:
        if args.calc_dir is not None:
            calc_location = proj_dir/args.calc_dir

    if calc_location is None:
        loc = check_location(proj_dir)
        calc_dir = config.get_dynamic_attr("{var}.calc_dir", loc)
        calc_location = proj_dir / (calc_dir + f'{second_suffix}')  #'calc_local_tmp'


    return calc_location

def set_output_path(args, calc_location, config):
    output_dir = None
    if args is not None:
        if args.output_dir is not None:
            output_dir = calc_location / args.output_dir

    if output_dir is None:
        try:
            output_dir = calc_location / config.hpc.output_dir
        except AttributeError:
            print('AttributeError: config.hpc.output_dir not found', file=sys.stdout, flush=True)


    return output_dir

def replace(template, d):
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
    return template

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

def set_grp_path(parent_path, d, config=None, source='csv', grp_level='grp_dir_structure', make_grp=True):
    if 'tp' in d.keys():
        d['Tp'] = d['tp']
    tmp_d = d.copy()
    tmp_d= {k:v[0] if isinstance(v, list) and len(v)==1 else v for k,v in tmp_d.items()}
    config_path = parent_path#set_model_config_path(parent_path, d, config=config)

    if source == 'csv':
        if 'lag' in d and d['lag'] is not None:
            grp_level = 'dir_structure_csv'
        else:
            grp_level = 'grp_dir_structure'
        # grp_level = 'dir_structure_csv'
    else:
        grp_level = 'dir_structure'

    if config is not None:
        try:
            grp_path_template = config.get_dynamic_attr("output.{var}", grp_level) # config.output.grp_dir_structure
            grp_path_template_filled = template_replace(grp_path_template,tmp_d, return_replaced=False)
            grp_path = config_path / grp_path_template_filled

        except:
            pass
    else:
        grp_path_template_filled = f'{d["col_var_id"]}_{d["target_var_id"]} / E{d["E"]}_tau{d["tau"]}'
        grp_path = config_path / grp_path_template_filled


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

    if config is not None:
        try:
            dir_structure = config.output.non_grp_structure
            dir_structure_filled = replace(dir_structure, d)
            return parent_path / dir_structure_filled
        except:
            pass
    return parent_path / f'knn_{d["knn"]}' / f'tp_{d["Tp"]}' / f'lag_{d["lag"]}'
