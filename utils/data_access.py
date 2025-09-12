from operator import index

import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import datetime
import gc
import re

def streamline_cause(label, split_word='causes'):
    label_parts = label.split(' {} '.format(split_word))
    clean_parts = []
    for part in label_parts:
        # print(part.split('(t-'))
        try:
            part = ' '.join(list(set([col.split(')')[-1].strip(' ') for col in part.split('(t-')]))).strip(' ')
        except:
            part = part.strip(' ')
        clean_parts.append(part)
    label = f' {split_word} '.join(clean_parts)
    return label


def remove_numbers(input_string):
    # Use regex to remove all digits from the string
    return re.sub(r'\d+', '', input_string)

def extract_numbers(input_string):
    # Use regex to extract all numbers from the string
    return re.findall(r'\d+', input_string)[0]  # Assuming numbers are before the first '__'

def relationship_filter(df, rel):
    return df[df['relation'].isin([rel, rel.replace('influences', 'causes')])].copy()

def rel_reformat(df, rel):
    if rel in df.columns:
        df[rel] = df[rel].str.replace('causes', 'influences')
    return df

# Function to fetch and prepare real data
def get_real_output(real_dfs_references, grp_path, meta_variables, max_libsize, knn, sample_size=500,verbose=False, kwargs=None):
    nrows = kwargs.get('nrows', None) if isinstance(kwargs, dict) is True else None
    if isinstance(real_dfs_references,list) ==True:
        if len(real_dfs_references) == 0:
            print('No real data found', file=sys.stderr, flush=True)
            return None
        elif len(real_dfs_references) > 1:
            # files = [real_dfs_references[0]]
            real_df_full = pd.concat([pd.read_csv(grp_path / file) for file in real_dfs_references])
            if verbose is True:
                print('Multiple files found, concatenating:', len(real_dfs_references), file=sys.stderr, flush=True)
        else:
            files = real_dfs_references
            real_df_full = pd.read_csv(grp_path / files[0], low_memory=False,
                                       usecols=['LibSize', 'rho', 'relation','E', 'tau', 'lag'],
                                       dtype={'LibSize': np.int32, 'relation': str,
                            'ind_i': int, 'rho': np.float64, 'tau': int, 'E': int,'Tp':int,
                            'pset_id': str, 'MAE': float, 'RMSE':float, 'forcing':str, 'responding':str})
                                       # , low_memory=True, usecols=['LibSize', 'rho', 'relation','E', 'tau', 'lag'],
                                       # dtype={'LibSize': np.int32, 'relation': str,
                            # 'weighted': bool, 'surr_var': str, 'surr_num': int, 'rho': float, 'tau': int, 'E': int,
                            # 'pset_id': str, 'MAE': float, 'RMSE':float, 'forcing':str, 'responding':str}, nrows=nrows)        # print(real_df_full.head(), file=sys.stderr, flush=True)
            # print(real_df_full.shape, file=sys.stderr, flush=True)
    else:
        real_df_full = collect_raw_output(real_dfs_references, meta_vars=meta_variables)

    real_df_full = real_df_full[real_df_full['LibSize'] >= knn].copy()
    if real_df_full.empty:
        return None

    real_df = real_df_full[real_df_full['LibSize'] <= max_libsize].copy()
    rel_dfs = [
        rel_df.groupby('LibSize').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
        for _, rel_df in real_df.groupby('relation')
    ]
    real_df = pd.concat(rel_dfs).reset_index(drop=True)
    real_df['surr_var'] = 'neither'
    real_df['surr_num'] = 0

    try:
        real_df['relation'] = real_df['relation'].apply(lambda x: streamline_cause(x))
    except Exception as e:
        print(f'Error applying streamline_cause: {e}', file=sys.stderr)

    return real_df


# Function to fetch and prepare surrogate data
def get_surrogate_output(surr_dfs_references, grp_path, meta_variables, max_libsize, knn, _rel,
                         sample_size=200, verbose=False, kwargs=None):

    kwargs = {} if kwargs is None else kwargs
    if 'surr_nums' in kwargs:
        surr_nums = kwargs['surr_nums']
    else:
        surr_nums = None
    # print('surr_nums', surr_nums, file=sys.stderr, flush=True)
    # print(f'kwargs: {kwargs}', file=sys.stderr, flush=True)
    # surr_nums = kwargs.get('surr_nums', None) if isinstance(kwargs, dict) is True else None
    # # print(surr_nums, file=sys.stderr, flush=True)
    surr_dfs = []
    ctr = 0
    for df_csv_name in surr_dfs_references:
        if isinstance(df_csv_name, str) is True:
            if 'neither' in df_csv_name:
                continue
        elif 'neither' in df_csv_name.name:
            continue

        try:
            surr_df_full = pd.read_csv(grp_path / df_csv_name,low_memory=True, )#, chunksize=5000, low_memory=True)
        except:
            continue
        surr_df = relationship_filter(surr_df_full, _rel) #surr_df_full[surr_df_full['relation'].isin([_rel, _rel.replace('influences', 'causes')])].copy()

        if surr_df.empty:
            if verbose is True:
                print(f'Empty surrogate data for {df_csv_name}', file=sys.stderr, flush=True)
            continue

        df_csv_name_str = df_csv_name if isinstance(df_csv_name, str) is True else df_csv_name.name
        surr_var = remove_numbers(df_csv_name_str.split('__')[1].split('.csv')[0])
        surr_num = int(extract_numbers(df_csv_name_str.split('__')[1].split('.csv')[0]))
        if surr_nums is not None:
            if surr_num not in surr_nums:
                # print(f'Skipping surrogate {surr_var} {surr_num} not in sur_nums', file=sys.stderr, flush=True)
                continue
        if surr_var == 'tsi':
            surr_var = 'TSI'
        surr_df['surr_var'] = surr_var
        surr_df['surr_num'] = int(surr_num)

        surr_df = surr_df[(surr_df['surr_var'] != 'neither') & (surr_df['LibSize'] >= knn) & (surr_df['LibSize'] <= max_libsize)].copy()
        rel_dfs = []
        for grp_rel, rel_df in surr_df.groupby('relation'):
            # print('rel', grp_rel, file=sys.stderr, flush=True)
            rel_dfs.append(rel_df.groupby('LibSize').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True))

        # print('attempt to concat', file=sys.stderr, flush=True)
        surr_df = pd.concat(rel_dfs).reset_index(drop=True)
        rel_dfs = []
        for surr_var, surr_df_i in surr_df.groupby('surr_var'):
            surr_df_i['relation_s'] = surr_df_i['relation'].str.replace(surr_var, f'{surr_var} (surr) ')
            surr_df_i['relation_s'] =surr_df_i['relation_s'].str.strip()
            surr_df_i['relation_s'] = surr_df_i['relation_s'].str.replace('  ', ' ')
            rel_dfs.append(surr_df_i)

        surr_df = pd.concat(rel_dfs).reset_index(drop=True)
        if len(surr_df)>0:
            surr_dfs.append(surr_df)
            ctr += 1

    if len(surr_dfs) == 0:
        # print('No surrogate data found', file=sys.stderr, flush=True)
        return None
    else:
        surr_df_full = pd.concat(surr_dfs).reset_index(drop=True)
        print('surr_df_full', surr_df_full.shape, file=sys.stderr, flush=True)
        return surr_df_full

def collect_raw_output(dfs_references, meta_vars= None):
    real_dfs = []
    for pset_id, real_pset_ref in dfs_references.groupby('pset_id'):
        real_pset_ref_d = {}
        if meta_vars is not None:
            real_pset_ref_d = real_pset_ref[meta_vars].iloc[0].to_dict()
        pset_dfs = []
        for df_path in real_pset_ref.df_path.values:
            if 'pctile' not in str(df_path):
                if Path(df_path).exists():
                    _real_df = pd.read_csv(df_path, index_col=0)
                    pset_dfs.append(_real_df)
        if len(pset_dfs) > 0:
            _real_df = pd.concat(pset_dfs)
            if len(real_pset_ref_d) > 0:
                for key, value in real_pset_ref_d.items():
                    _real_df[key] = value
            real_dfs.append(_real_df)

    columns = list(dfs_references.columns)
    if 'relation' not in columns:
        columns.append('relation')

    real_df = pd.DataFrame(columns=columns)
    if len(real_dfs) > 0:
        real_df = pd.concat(real_dfs)

    return real_df


def get_weighted_flag(grp_d):
    if grp_d['weighted'] in ['True', True]:
        weighted_flag = 'weighted'
    elif grp_d['weighted'] in ['False', False]:
        weighted_flag = 'unweighted'
    return weighted_flag

def set_df_weighted(grp_df, weighted_flag):
    if weighted_flag == 'weighted':
        grp_df = grp_df[grp_df['weighted'] == True].copy()
    elif weighted_flag == 'unweighted':
        grp_df = grp_df[grp_df['weighted'] == False].copy()
    return grp_df




def check_for_exit(override_flag, datetime_flag, metric_df_path, grp_d):

    choice = 'continue'
    # Check if the metric dataframe already exists and if it should be overridden
    if (override_flag is False) and (datetime_flag is not None):
        if metric_df_path.exists() is True:
            # file modification timestamp of a file
            m_time = os.path.getmtime(metric_df_path)
            # convert timestamp into DateTime object
            dt_m = datetime.datetime.fromtimestamp(m_time)
            print('m_time', m_time, 'dt_m', dt_m, 'datetime_flag', datetime_flag, file=sys.stdout, flush=True)
            try:
                if dt_m < datetime_flag:
                    print(
                        f'datetime flag--> proceed for grp_id:{str(grp_d["group_id"])} already exists, datetime flag: {datetime_flag}, modification timestamp:{dt_m}',
                        file=sys.stdout, flush=True)
                else:
                    print(
                        f'datetime flag--> halt for grp_id:{str(grp_d["group_id"])} already exists, datetime flag: {datetime_flag}, modification timestamp:{dt_m}',
                        file=sys.stdout, flush=True)
                    choice = 'exit'
                    return choice
            except:
                print(f'error with datetime flag for grp_id:{str(grp_d["group_id"])}', 'will continue', file=sys.stdout,
                      flush=True)
    else:
        if (override_flag == False) and (metric_df_path.exists() is True):
            print(f'grp_id:{str(grp_d["group_id"])} already exists', 'exiting: no datetime comparison requested',
                  file=sys.stdout, flush=True)
            choice = 'exit'
            return choice
    return choice


def write_query_string(query_keys, grp_d):
    query_d = {}
    for key in query_keys:
        if isinstance(grp_d[key], str):
            query_d[key] = f'"{grp_d[key]}"'
        else:
            query_d[key] = grp_d[key]

    query_string = [f'{key}=={value}' for key, value in query_d.items()]
    query_string = ' & '.join(query_string)

    return query_string


def pull_raw_data(config, proj_dir, var_ids, alias=True):

    time_var = config.raw_data.time_var
    time_var_alias = 'date'#config.raw_data.time_var_alias

    data_dfs = []
    var_aliases = []
    for var_id in var_ids:
        data_var = config.get_dynamic_attr("{var}.data_var", var_id)
        if alias == True:
            data_var_alias = config.get_dynamic_attr("{var}.var", var_id)
        else:
            data_var_alias = data_var
        var_aliases.append(data_var_alias)

        # try:
        var_data_csv = config.get_dynamic_attr("{var}.data_csv", var_id)
        print(f'Pulling raw data for {var_id} from {var_data_csv}', file=sys.stdout, flush=True)
        # except:
        #     var_data_csv = config.raw_data.data_csv
        var_data = pd.read_csv(proj_dir / config.raw_data.name / f'{var_data_csv}.csv')
        var_data = var_data[[time_var, data_var]].rename(columns=
                                                                         {time_var: time_var_alias,
                                                                          data_var: data_var_alias,
                                                                          })
        var_data = var_data.dropna(subset=[data_var_alias], how='all')
        var_data = var_data[[time_var_alias, data_var_alias]].copy()
        data_dfs.append(var_data)

    data = data_dfs[0]
    for var_df in data_dfs[1:]:
        data = pd.merge(data, var_df, on=time_var_alias, how='outer')

    data = data.dropna(subset=var_aliases, how='all')
    # print(f'Raw data shape: {data.shape}, {data.head}', file=sys.stdout, flush=True)
    return data


def check_empty_concat(lst):
    if len(lst) > 0:
        return pd.concat(lst)
    else:
        return pd.DataFrame()