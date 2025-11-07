import os
import sys
from pathlib import Path
import re
import pandas as pd
import pyEDM as pe
import yaml
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.data_access import pull_raw_data

from utils.location_helpers import *
import copy
#
# # python3 carc2/make_surrogates.py --project $PROJECT -n 3


def increment_var_name(col_name, increment, prefix):
    if col_name.startswith(prefix):
        num = int(col_name.split('_')[1])
        return f'{prefix}_{num + increment}'
    return col_name


def process_group(arg_tuple):
    var_id, data_var, var_alias, num_surrogates, surr_group, proj_path, df = arg_tuple
    print(f'var_id: {var_id}, data_var: {data_var}, var_alias: {var_alias}, len: {len(df)}', file=sys.stdout, flush=True)
    print(df.columns, file=sys.stdout, flush=True)
    csv_file_name = f'pe_{surr_group}_surr_{var_id}_{var_alias}.csv'

    start_index = 0 if len(df) % 2 == 0 else 1

    dfs = []
    pe_surr_i = pe.SurrogateData(dataFrame=df.iloc[start_index:], column=var_alias, numSurrogates=num_surrogates)
    pe_surr_i = pe_surr_i[[col for col in pe_surr_i.columns if col != var_alias]]

    pattern = r'\d+'
    surr_path = proj_path / 'surrogates' / csv_file_name
    print('surr_path', surr_path, file=sys.stdout, flush=True)
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
    # if spec_config and spec_config.has_nested_attribute("raw_data.data_csv"):
    #     data_csv = spec_config.raw_data.data_csv
    #     time_var = spec_config.raw_data.time_var
    #     data_df = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)
    # else:
    #     data_csv = config.raw_data.data_csv
    #     time_var = config.raw_data.time_var
    #     data_df = pd.read_csv(proj_dir / config.raw_data.name / f'{data_csv}.csv', index_col=0)

    surr_arg_tuples = []
    if args.vars not in [None, '', ['None'], 'None', [None]]:
        var_ids = args.vars
        print('vars', args.vars, file=sys.stdout, flush=True)
    else:
        var_ids = copy.deepcopy(config.col.ids) + config.target.ids

    print('var_ids', var_ids, file=sys.stdout, flush=True)
    for var_id in var_ids:
        var_conf = getattr(config, var_id)
        data_csv = var_conf.data_csv
        # time_var = spec_config.raw_data.time_var
        data_df = pd.read_csv(proj_dir / 'data' / f'{data_csv}.csv', index_col=0)

        data_var = var_conf.data_var
        var_alias = var_conf.var
        data_df = pull_raw_data(config, proj_dir, [var_id])
        # tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
        surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, data_df))

    # elif spec_config:
    #     for var_type in ['col_var', 'target_var']:
    #         if hasattr(spec_config, var_type):
    #             var_ids = getattr(spec_config, var_type).id
    #             for var_id in var_ids:
    #                 var_conf = getattr(spec_config, var_id)
    #                 data_var = var_conf.data_var
    #                 var_alias = var_conf.var
    #                 data_df = pull_raw_data(spec_config, proj_dir, [var_id])
    #                 tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
    #             surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))

    # else:
    #     var_ids = copy.deepcopy(config.col.ids) + config.target.ids
    #     for var_id in var_ids:
    #         var_conf = getattr(config, var_id)
    #         data_var = var_conf.data_var
    #         var_alias = var_conf.var
    #         print('var_id', var_id, 'data_var', data_var, 'var_alias', var_alias, file=sys.stdout, flush=True)
    #         tmp_data_df = data_df.rename(columns={time_var: 'date', data_var: var_alias})
    #         surr_arg_tuples.append((var_id, data_var, var_alias, num_surrogates, surr_group, proj_dir, tmp_data_df))

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
            existing = [csv_file]
        setattr(var_conf, 'surr_file_name', existing)# else [existing, csv_file])
        config.save_config()


        # Update spec config if applicable
        if spec_config:
            var_conf_spec = getattr(spec_config, var_id)
            existing_spec = getattr(var_conf_spec, 'surr_file_name', None)
            if isinstance(existing_spec, list):
                if csv_file not in existing_spec:
                    existing_spec.append(csv_file)
            else:
                existing_spec = [csv_file]
            setattr(var_conf_spec, 'surr_file_name', existing_spec)# else [existing_spec, csv_file])

            # Save updates
            if spec_config_path:
                with open(spec_config_path, 'w') as f:
                    yaml.dump(spec_config.to_dict(), f)
