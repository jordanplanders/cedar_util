import os
import sys
from pathlib import Path
import re
import pandas as pd
import pyEDM as pe
import yaml
from utils.arg_parser import get_parser
from utils.config_parser import load_config

from utils.location_helpers import *
import copy
#
# # python3 carc2/make_surrogates.py --project $PROJECT -n 3
#
# def increment_var_name(col_name, increment, prefix):
#     if col_name.startswith(prefix):
#         num = int(col_name.split('_')[1])
#         return f'{prefix}_{num + increment}'
#     return col_name
#
#
# def process_group(arg_tuple):
#     var_id, data_var, var_alias, num_surrogates, surr_group, proj_path, df = arg_tuple
#     print(f'var_id: {var_id}, data_var: {data_var}, var_alias: {var_alias}', file=sys.stdout, flush=True)
#
#     csv_file_name = f'pe_{surr_group}_surr_{var_id}_{var_alias}.csv'
#
#     if len(df) % 2 == 0:
#         start_index = 0
#     else:
#         start_index = 1
#
#     dfs = []
#     pe_surr_i = pe.SurrogateData(dataFrame=df.iloc[start_index:], column=var_alias,
#                                  numSurrogates=num_surrogates)
#     pe_surr_i = pe_surr_i[[col for col in pe_surr_i.columns if col not in [var_alias]]]
#     # pe_surr_i[var_alias] = df[var][1:]
#
#     pattern = r'\d+'
#     if (proj_path / 'surrogates' / csv_file_name).exists():
#         existing_surrs = pd.read_csv(proj_path / 'surrogates' / csv_file_name, index_col=0)
#         dfs.append(existing_surrs)
#         existing_surrs_final_ind = max(
#             [int(re.findall(pattern, col)[0]) for col in existing_surrs.columns if len(re.findall(pattern, col)) > 0])
#         revised_cols = {col: increment_var_name(col, existing_surrs_final_ind, var_alias) for col in pe_surr_i.columns}
#         pe_surr_i.rename(columns=revised_cols, inplace=True)
#         pe_surr_i = pe_surr_i[[col for col in pe_surr_i.columns if col not in existing_surrs.columns]]
#
#     dfs.append(pe_surr_i)
#
#     pe_surr_all = pd.concat(dfs, axis=1)
#     pe_surr_all.drop_duplicates(inplace=True)
#
#     pe_surr_all.to_csv(proj_path / 'surrogates' / csv_file_name)
#
#     return csv_file_name
#
#
# if __name__ == '__main__':
#     # grab parameter file
#     parser = get_parser()
#     args = parser.parse_args()
#
#     # Example usage of arguments
#     # print(f"Number from script2: {args.number}")
#     if 'project' in args:
#         proj_name = args.project
#     else:
#         print('project name is required', file=sys.stdout, flush=True)
#         print('project name is required', file=sys.stderr, flush=True)
#         sys.exit(0)
#
#     # proj_dir = Path(os.getcwd()) / proj_name
#     proj_dir = set_proj_dir(proj_name, os.getcwd())
#     with open(proj_dir / 'proj_config.yaml', 'r') as file:
#         config = yaml.safe_load(file)
#
#     (proj_dir / 'surrogates').mkdir(parents=True, exist_ok=True)
#
#     if 'number' in args:
#         num_surrogates = args.number
#     else:
#         num_surrogates = 150
#
#     if args.group is not None:
#         surr_group = args.group
#     else:
#         surr_group = proj_name
#
#     spec_config = {}
#     spec_config_yml = None
#     if args.config is not None:
#         spec_config_yml = args.config
#         with open(proj_dir / f'{spec_config_yml}.yaml', 'r') as file:
#             spec_config = yaml.safe_load(file)
#
#     if 'data_csv' in spec_config:
#         data_csv = spec_config['raw_data']['data_csv']
#         time_var = spec_config['raw_data']['time_var']
#         data_df = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)
#     else:
#         data_csv = config.raw_data.data_csv
#         time_var = config.raw_data.time_var
#         data_df = pd.read_csv(proj_dir / config.raw_data.name / f'{data_csv}.csv', index_col=0)
#
#     surr_arg_tuples = []
#     if args.vars is not None:
#     # if 'variable' in args:
#         for var_id in args.vars:
#         # var_id = args.vars
#             if var_id in config:
#                 data_var = config[var_id]['data_var']
#                 var_alias = config[var_id]['var']
#                 tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
#
#                 surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))
#
#     elif len(spec_config) > 0:
#         for var_type in ['col_var', 'target_var']:
#             if var_type in spec_config:
#                 var_id = spec_config[var_type]['id']
#                 data_var = spec_config[var_id]['data_var']
#                 var_alias = spec_config[var_id]['var']
#                 tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
#
#                 surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))
#
#     else:
#         var_ids = copy.copy(config.col.ids)
#         var_ids.extend(config.target.ids)
#         for var_id in var_ids:
#             data_var = getattr(config, f"{var_id}.data_var")
#             var_alias = getattr(config, f"{var_id}.var")
#             # data_var = config[var_id]['data_var']
#             # var_alias = config[var_id]['var']
#             tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
#
#             surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))
#
#     for arg_tuple in surr_arg_tuples:
#         csv_file = process_group(arg_tuple)
#         var_id = arg_tuple[0]
#
#         if type(config[var_id]['surr_file_name']) == list:
#             config[var_id]['surr_file_name'].append(csv_file)
#         else:
#             config[var_id]['surr_file_name'] = csv_file
#
#         if len(spec_config) > 0:
#             if type(spec_config[var_id]['surr_file_name']) == list:
#                 spec_config[var_id]['surr_file_name'].append(csv_file)
#             else:
#                 spec_config[var_id]['surr_file_name'] = csv_file
#
#     with open(proj_dir / 'proj_config.yaml', 'w') as file:
#         yaml.dump(config, file)
#
#     if spec_config_yml is not None:
#         with open(proj_dir / f'{spec_config_yml}.yaml', 'w') as file:
#             yaml.dump(spec_config, file)


# import os
# import sys
# from pathlib import Path
# import re
# import pandas as pd
# import pyEDM as pe
# import yaml
# import copy
#
# from utils.arg_parser import get_parser
# from utils.location_helpers import *
# from utils.project_config import load_config, ProjectConfig  # assuming you moved class to utils

def increment_var_name(col_name, increment, prefix):
    if col_name.startswith(prefix):
        num = int(col_name.split('_')[1])
        return f'{prefix}_{num + increment}'
    return col_name


def process_group(arg_tuple):
    var_id, data_var, var_alias, num_surrogates, surr_group, proj_path, df = arg_tuple
    print(f'var_id: {var_id}, data_var: {data_var}, var_alias: {var_alias}', file=sys.stdout, flush=True)

    csv_file_name = f'pe_{surr_group}_surr_{var_id}_{var_alias}.csv'

    start_index = 0 if len(df) % 2 == 0 else 1

    dfs = []
    pe_surr_i = pe.SurrogateData(dataFrame=df.iloc[start_index:], column=var_alias, numSurrogates=num_surrogates)
    pe_surr_i = pe_surr_i[[col for col in pe_surr_i.columns if col != var_alias]]

    pattern = r'\d+'
    surr_path = proj_path / 'surrogates' / csv_file_name
    if surr_path.exists():
        existing_surrs = pd.read_csv(surr_path, index_col=0)
        dfs.append(existing_surrs)

        existing_surrs_final_ind = max(
            [int(re.findall(pattern, col)[0]) for col in existing_surrs.columns if re.findall(pattern, col)]
        )
        revised_cols = {col: increment_var_name(col, existing_surrs_final_ind, var_alias) for col in pe_surr_i.columns}
        pe_surr_i.rename(columns=revised_cols, inplace=True)
        pe_surr_i = pe_surr_i[[col for col in pe_surr_i.columns if col not in existing_surrs.columns]]

    dfs.append(pe_surr_i)

    pe_surr_all = pd.concat(dfs, axis=1)
    pe_surr_all.drop_duplicates(inplace=True)
    pe_surr_all.to_csv(surr_path)

    return csv_file_name


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not hasattr(args, 'project') or args.project is None:
        print('project name is required', file=sys.stderr)
        sys.exit(1)

    proj_name = args.project
    proj_dir = set_proj_dir(proj_name, Path(os.getcwd()))
    config = load_config(proj_dir / 'proj_config.yaml')
    # config.file_path = proj_dir / 'proj_config.yaml'


    # with open(proj_dir / 'proj_config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # config = load_config(proj_dir / 'proj_config.yaml')

    (proj_dir / 'surrogates').mkdir(parents=True, exist_ok=True)

    num_surrogates = getattr(args, 'number', 150)
    surr_group = getattr(args, 'group', proj_name)

    # Handle optional specific config
    spec_config = None
    spec_config_path = None
    if args.config:
        spec_config_path = proj_dir / f'{args.config}.yaml'
        spec_config = load_config(spec_config_path)

    # Load data
    if spec_config and spec_config.has_nested_attribute("raw_data.data_csv"):
        data_csv = spec_config.raw_data.data_csv
        time_var = spec_config.raw_data.time_var
        data_df = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)
    else:
        data_csv = config.raw_data.data_csv
        time_var = config.raw_data.time_var
        data_df = pd.read_csv(proj_dir / config.raw_data.name / f'{data_csv}.csv', index_col=0)

    surr_arg_tuples = []

    if args.vars:
        for var_id in args.vars:
            var_conf = getattr(config, var_id)
            data_var = var_conf.data_var
            var_alias = var_conf.var
            tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
            surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))

    elif spec_config:
        for var_type in ['col_var', 'target_var']:
            if hasattr(spec_config, var_type):
                var_id = getattr(spec_config, var_type).id
                var_conf = getattr(spec_config, var_id)
                data_var = var_conf.data_var
                var_alias = var_conf.var
                tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
                surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))

    else:
        var_ids = copy.deepcopy(config.col.ids) + config.target.ids
        for var_id in var_ids:
            var_conf = getattr(config, var_id)
            data_var = var_conf.data_var
            var_alias = var_conf.var
            tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
            surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))

    for arg_tuple in surr_arg_tuples:
        csv_file = process_group(arg_tuple)
        var_id = arg_tuple[0]

        # Update main config
        var_conf = getattr(config, var_id)
        existing = getattr(var_conf, 'surr_file_name', None)
        if isinstance(existing, list):
            if csv_file not in existing:
                existing.append(csv_file)
        else:
            setattr(var_conf, 'surr_file_name', [csv_file])# else [existing, csv_file])

        # Update spec config if applicable
        if spec_config:
            var_conf_spec = getattr(spec_config, var_id)
            existing_spec = getattr(var_conf_spec, 'surr_file_name', None)
            if isinstance(existing_spec, list):
                if csv_file not in existing_spec:
                    existing_spec.append(csv_file)
            else:
                setattr(var_conf_spec, 'surr_file_name', [csv_file])# else [existing_spec, csv_file])

    # Save updates
    config.save_config()
    if spec_config_path:
        with open(spec_config_path, 'w') as f:
            yaml.dump(spec_config.to_dict(), f)
