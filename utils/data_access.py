import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import datetime

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

def relationship_filter(df, rel):
    return df[df['relation'].isin([rel, rel.replace('influences', 'causes')])].copy()

def rel_reformat(df, rel):
    if rel in df.columns:
        df[rel] = df[rel].str.replace('causes', 'influences')
    return df

# Function to fetch and prepare real data
def get_real_data(real_dfs_references, grp_path, meta_variables, max_libsize, knn, sample_size=500):
    if isinstance(real_dfs_references,list) ==True:
        if len(real_dfs_references) == 0:
            print('No real data found', file=sys.stderr, flush=True)
            return None
        elif len(real_dfs_references) > 1:
            # files = [real_dfs_references[0]]
            real_df_full = pd.concat([pd.read_csv(grp_path / file) for file in real_dfs_references])
            print('Multiple files found, concatenating:', len(real_dfs_references), file=sys.stderr, flush=True)
        else:
            files = real_dfs_references
            real_df_full = pd.read_csv(grp_path / files[0])
        # print(real_df_full.head(), file=sys.stderr, flush=True)
    else:
        real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)

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

    try:
        real_df['relation'] = real_df['relation'].apply(lambda x: streamline_cause(x))
    except Exception as e:
        print(f'Error applying streamline_cause: {e}', file=sys.stderr)

    return real_df


# Function to fetch and prepare surrogate data
def get_surrogate_data(surr_dfs_references, grp_path, meta_variables, max_libsize, knn, _rel,
                       sample_size=400):
    surr_dfs = []
    ctr = 0
    for df_csv_name in surr_dfs_references:
        surr_df_full = pd.read_csv(grp_path / df_csv_name)
        surr_df = relationship_filter(surr_df_full, _rel) #surr_df_full[surr_df_full['relation'].isin([_rel, _rel.replace('influences', 'causes')])].copy()

        if surr_df.empty:
            continue

        if 'surr_var' not in surr_df.columns:
            surr_var = remove_numbers(df_csv_name.split('__')[1].split('.csv')[0])
            if surr_var == 'tsi':
                surr_var = 'TSI'
            surr_df['surr_var'] = surr_var

        surr_df = surr_df[(surr_df['surr_var'] != 'neither') & (surr_df['LibSize'] >= knn) & (surr_df['LibSize'] <= max_libsize)].copy()
        rel_dfs = [
            rel_df.groupby('LibSize').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
            for _, rel_df in surr_df.groupby('surr_var')
        ]
        surr_df = pd.concat(rel_dfs).reset_index(drop=True)
        rel_dfs = []
        for surr_var, surr_df_i in surr_df.groupby('surr_var'):
            surr_df_i['relation_s'] = surr_df_i['relation'].str.replace(surr_var, f'{surr_var} (surr) ')
            surr_df_i['relation_s'] =surr_df_i['relation_s'].str.strip()
            surr_df_i['relation_s'] = surr_df_i['relation_s'].str.replace('  ', ' ')
            rel_dfs.append(surr_df_i)
        surr_df = pd.concat(rel_dfs).reset_index(drop=True)
        # surr_df['relation_s'] = surr_df.apply(lambda row: row['relation'].replace(row['surr_var'], f'{row["surr_var"]} (surr)'), axis=1)
        if len(surr_df)>0:
            surr_dfs.append(surr_df)
            ctr += 1
        # if ctr > 3:
        #     continue

    if len(surr_dfs) == 0:
        print('No surrogate data found', file=sys.stderr, flush=True)
        return None
    else:
        surr_df_full = pd.concat(surr_dfs).reset_index(drop=True)
        return surr_df_full

def collect_raw_data(dfs_references, meta_vars= None):
    real_dfs = []
    # print(grp_d, file=sys.stdout, flush=True)
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


def pull_percentile_data(df, filter_var='rho', groupby_var = 'LibSize',
                         percentiles = None, sample_kwargs = None):

    if percentiles is None:
        percentiles = [0.25, 0.75]

    if type(percentiles[0]) not in [float]:
        print('percentiles should be float')
        percentiles = [float(i) for i in percentiles]

    if sample_kwargs is None:
        sample_kwargs = {'n': 1}

    if 'replace' not in sample_kwargs:
        sample_kwargs['replace'] = False

    flag = ''
    iqr_list = []
    for libsize, libsize_df in df.groupby(groupby_var):
        # libsize_df = libsize_df.dropna()

        rho_l = libsize_df[filter_var].quantile(percentiles[0])
        rho_u = libsize_df[filter_var].quantile(percentiles[1])
        min_n = []
        _df_sub = libsize_df[(libsize_df[filter_var] >= rho_l) & (libsize_df[filter_var] <= rho_u)].copy()
        if len(_df_sub) <1:
            continue

        if 'n' in sample_kwargs:
            if sample_kwargs['replace'] is True:
                if len(_df_sub)/sample_kwargs['n']<.3:
                    sample_kwargs['n'] = int(len(_df_sub)/.3)
                    flag = 'n'
            else:
                if len(_df_sub) < sample_kwargs['n']:
                    sample_kwargs['n'] = len(_df_sub)
                    flag = 'n'
                # print(f'libsize:{libsize}, rho_l:{rho_l}, rho_u:{rho_u}, n:{len(_df_sub)}')
        if len(_df_sub) > 0:
            iqr_list.append(_df_sub.sample(**sample_kwargs))
            min_n.append(len(iqr_list[-1]))

    if len(min_n) > 0:
        min_n = min(min_n)
    else:
        min_n = 0

    if len(iqr_list) > 0:
        libsize_sub = pd.concat(iqr_list)
        if flag !='n':
            flag = 'pass'
    else:
        libsize_sub = df
        flag = 'no_sampling'
    return libsize_sub, flag, min_n


def get_group_sizes(df, filter_var='rho', groupby_var = 'LibSize',
                         percentiles = None, sample_kwargs = None):

    if percentiles is None:
        percentiles = [0.25, 0.75]

    if type(percentiles[0]) not in [float]:
        print('percentiles should be float')
        percentiles = [float(i) for i in percentiles]

    iqr_list = []
    for libsize, libsize_df in df.groupby(groupby_var):
        # libsize_df = libsize_df.dropna()

        rho_l = libsize_df[filter_var].quantile(percentiles[0])
        rho_u = libsize_df[filter_var].quantile(percentiles[1])
        iqr_list.append(len(libsize_df[(libsize_df[filter_var] >= rho_l) & (libsize_df[filter_var] <= rho_u)]))

    return iqr_list

def get_sample_rep_n(libize_grp_sizes):
    if min(libize_grp_sizes) < 200:
        sample_n = int(.3 * min(libize_grp_sizes))
        rep_times = 7
    elif min(libize_grp_sizes) < 1000:
        sample_n = int(.5 * min(libize_grp_sizes))
        rep_times = 5
    else:
        sample_n = min(int(.7 * min(libize_grp_sizes)), 800)
        rep_times = 3
    return sample_n, rep_times


def bootstrap_raw_output(raw_df_full, filter_var, groupby_var, pctile_range, sample_kwargs=None, grp_d=None):
    libize_grp_sizes = get_group_sizes(raw_df_full, filter_var=filter_var,
                                       groupby_var=groupby_var, percentiles=pctile_range)

    sample_n, rep_times = get_sample_rep_n(libize_grp_sizes)
    if sample_kwargs is None:
        sample_kwargs = {'replace': False}
    sample_kwargs['n'] = sample_n
    if grp_d is None:
        grp_d = {'group_id':'regression sampling'}#raw_df_full['group_id'].iloc[0]}

    sampling_errors = []
    samples = []
    for _ in range(rep_times):
        sampling_df, sampling_flag, min_n = pull_percentile_data(raw_df_full, filter_var=filter_var,
                                                                      groupby_var=groupby_var,
                                                                      percentiles=pctile_range, sample_kwargs=sample_kwargs)

        if sampling_flag != 'pass':
            flag_text = f'issue sampling real {grp_d["group_id"]}: {sampling_flag}, min_n={min_n}, specified min_n={min(libize_grp_sizes)}, sample_n={sample_n}, reps={rep_times}'
            if flag_text not in sampling_errors:
                sampling_errors.append(flag_text)

        samples.append(sampling_df)
    return samples, sampling_errors



def check_empty_concat(lst):
    if len(lst) > 0:
        return pd.concat(lst)
    else:
        return pd.DataFrame()