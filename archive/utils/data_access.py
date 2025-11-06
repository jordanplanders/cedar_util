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
    """
    Simplifies a relationship label by removing time lag information.
    Args:
        label (str): The relationship label to be simplified.
        split_word (str): The word used to split the label (default is 'causes').
    Returns:
        str: The simplified relationship label.
    Notes:
        1. The function splits the label by the specified split_word.
        2. For each part, it removes any time lag information formatted as '(t-<number>)'.
        3. It then rejoins the cleaned parts with the split_word to form the final label.
    """
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
    return re.findall(r'\d+', input_string)[-1]  # Assuming numbers are before the first '__'

def relationship_filter(df, rel):
    """
    Filters the DataFrame to include only rows where the 'relation' column matches the specified relation
    or its 'causes' equivalent.
    Args:
        df (pd.DataFrame): The input DataFrame.
        rel (str): The relation to filter by.
    Returns:
        pd.DataFrame: The filtered DataFrame.

    Notes:
        1. If the specified relation contains 'influences', it also considers the equivalent 'causes' relation.
        2. If the specified relation contains 'causes', it also considers the equivalent 'influences' relation.
        3. If the specified relation does not contain either, it only filters by the exact relation.
    """

    return df[df['relation'].isin([rel, rel.replace('influences', 'causes')])].copy()

def rel_reformat(df, rel):
    """
    Replaces 'causes' with 'influences' in the specified relation column of the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        rel (str): The name of the relation column to modify.
    Returns:
        pd.DataFrame: The modified DataFrame with updated relation values.
    """

    if rel in df.columns:
        df[rel] = df[rel].str.replace('causes', 'influences')
    return df

# Function to fetch and prepare real data
def get_real_output(real_dfs_references, grp_path, meta_variables, max_libsize, knn, sample_size=500,verbose=False, kwargs=None):
    """
    Fetches and prepares output for real runs from the provided references.
    Args:
        real_dfs_references (list or pd.DataFrame): List of file names or DataFrame references for real data.
        grp_path (Path): Path to the group directory.
        meta_variables (list): List of metadata variables to include.
        max_libsize (int): Maximum library size to filter data.
        knn (int): Minimum library size to filter data.
        sample_size (int): Number of samples to draw per library size and relation. Default is 500.
        verbose (bool): If True, prints additional information. Default is False.
        kwargs (dict): Additional keyword arguments. Can include 'nrows' to limit rows read from CSV files.
    Returns:
        pd.DataFrame or None: Prepared DataFrame for real data or None if no valid data found.
    Notes:
        1. If real_dfs_references is a list of file names, it reads and concatenates them into a single DataFrame.
        2. If real_dfs_references is a DataFrame, it uses the collect_raw_output function to gather data.
        3. Filters the data based on library size (knn to max_libsize).
        4. Samples the data for each relation and library size.
        5. Adds 'surr_var' and 'surr_num' columns with default values.
        6. Applies streamline_cause function to the 'relation' column.
        7. Returns None if no valid data is found after filtering.

    Calls:
        collect_raw_output(real_dfs_references, meta_vars=meta_variables)
        streamline_cause(x)
        remove_numbers(df_csv_name_str.split('__')[1].split('.csv')[0])


    """

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
    else:
        real_df_full = collect_raw_output(real_dfs_references, meta_vars=meta_variables)

    if real_df_full.empty:
        if verbose is True:
            print('Empty real data after collection', file=sys.stderr, flush=True)
        return None
    # if len(knn)>1:
    #     print('knn should be a single integer value', file=sys.stderr, flush=True)
    #     return None
    print('knn', knn)
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
        surr_suffix = df_csv_name_str.split('__')[1].split('.csv')[0]
        surr_num = int(extract_numbers(surr_suffix))
        surr_var = surr_suffix.rsplit(str(surr_num), 1)[0]#remove_numbers(df_csv_name_str.split('__')[1].split('.csv')[0])

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
    """
    Collects and concatenates raw output data from multiple DataFrame references.
    Args:
        dfs_references (pd.DataFrame): DataFrame containing references to raw data files.
        meta_vars (list or None): List of metadata variables to include. Default is None.
    Returns:
        pd.DataFrame: Concatenated DataFrame containing raw output data.
    Notes:
        1. The function groups the input DataFrame by 'pset_id'.
        2. For each group, it reads the data from the specified file paths, excluding those containing 'pctile'.
        3. It concatenates the data from all file paths within a group.
        4. If metadata variables are provided, it adds them to the concatenated DataFrame.
        5. Finally, it concatenates all group DataFrames into a single DataFrame and returns it.
        6. If no valid data is found, it returns an empty DataFrame with the appropriate
              columns.
    Calls:
        None

    """
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
    """
    Determines the weighted flag based on the 'weighted' key in the provided dictionary.
    Args:
        grp_d (dict): Dictionary containing the 'weighted' key.
    Returns:
        str: 'weighted' if the 'weighted' key is True, 'unweighted' if False.
    """

    if grp_d['weighted'] in ['True', True]:
        weighted_flag = 'weighted'
    elif grp_d['weighted'] in ['False', False]:
        weighted_flag = 'unweighted'
    return weighted_flag

def set_df_weighted(grp_df, weighted_flag):
    """
    Filters the DataFrame based on the specified weighted flag.
    Args:
        grp_df (pd.DataFrame): The input DataFrame containing a 'weighted' column.
        weighted_flag (str): The weighted flag to filter by ('weighted' or 'unweighted').
    Returns:
        pd.DataFrame: The filtered DataFrame based on the weighted flag.
    """
    if weighted_flag == 'weighted':
        grp_df = grp_df[grp_df['weighted'] == True].copy()
    elif weighted_flag == 'unweighted':
        grp_df = grp_df[grp_df['weighted'] == False].copy()
    return grp_df

def remove_extra_index(df):
    """
    Removes the 'Unnamed: 0' column from the DataFrame if it exists.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

def check_time_var(df, time_var):
    """
    Checks and adjusts the time variable in the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        time_var (str): The time variable to check ('date' or 'time').
    Returns:
        str: The adjusted time variable based on the DataFrame's columns.
    Notes:
        1. If time_var is 'date' and 'time' exists in the DataFrame, it switches to 'time'.
        2. If time_var is 'time' and 'date' exists in the DataFrame, it switches to 'date'.
        3. Returns the original time_var if no changes are made.
    """
    if time_var == 'date':
        if 'time' in df.columns:
            time_var = 'time'
    elif time_var == 'time':
        if 'date' in df.columns:
            time_var = 'date'
    return time_var

def check_for_exit(override_flag, datetime_flag, metric_df_path, grp_d):
    """
    Determines whether to continue or exit based on the existence of a metric DataFrame and a datetime flag.
    Args:
        override_flag (bool): If True, overrides existing files.
        datetime_flag (datetime or None): A datetime cutoff for file modification.
        metric_df_path (Path): Path to the metric DataFrame file.
        grp_d (dict): Dictionary containing group information, including 'group_id'.
    Returns:
        str: 'continue' to proceed or 'exit' to halt further processing.
    Notes:
        1. If override_flag is False and datetime_flag is provided, it checks the file's modification time.
        2. If the file exists and its modification time is newer than datetime_flag, it returns 'exit'.
        3. If override_flag is False and datetime_flag is None, it checks if the file exists and returns 'exit' if it does.
        4. Otherwise, it returns 'continue'.
    Calls:
        None
    """

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
    """
    Constructs a query string for filtering a DataFrame based on specified keys and their values in a dictionary.
    Args:
        query_keys (list): List of keys to include in the query.
        grp_d (dict): Dictionary containing key-value pairs for the query.
    Returns:
        str: A query string formatted for DataFrame filtering.
    Notes:
        1. The function iterates over the provided keys and retrieves their corresponding values from the dictionary.
        2. If a value is a string, it is enclosed in double quotes for proper query formatting.
        3. The key-value pairs are then combined into a single query string using ' & ' as a separator.
    """
    query_d = {}
    for key in query_keys:
        if isinstance(grp_d[key], str):
            query_d[key] = f'"{grp_d[key]}"'
        else:
            query_d[key] = grp_d[key]

    query_string = [f'{key}=={value}' for key, value in query_d.items()]
    query_string = ' & '.join(query_string)

    return query_string

def choose_data_source(proj_dir, config, data_source, var_data_csv=None, data_type='raw'):
    """
    Chooses the data source based on the specified data_source parameter.
    Args:
        proj_dir (Path): The project directory path.
        config (Config): The configuration object containing data source information.
        data_source (str): The data source to use ('data', 'raw_data', 'master_data', 'master').
        var_data_csv (str): The CSV file name for the variable data.
    Returns:
        Tuple[Path, pd.DataFrame]: The path to the data file and the loaded DataFrame.

    Notes:
        1. If data_source is 'data' or 'raw_data', it first tries to read from config.raw_data.name directory.
        2. If data_source is 'master_data' or 'master', it first tries to read from the parent directory's 'master_data' directory.
        3. If the file is not found in the first location, it attempts to read from the alternative location.
        4. If the file is not found in either location, it returns None for both path and DataFrame.

    Calls:
        remove_extra_index(var_data)
    """

    if data_type == 'raw':
        try:
            config_source = config.raw_data.name
        except:
            config_source = 'data'
        alternative_source = 'master_data'
    elif data_type in ['surr', 'surrogate']:
        try:
            config_source = config.surrogate_data.name
        except:
            config_source = 'surrogates'
        alternative_source = 'master_surrogates'

    data_path, var_data = None, None

    if 'master' in data_source:
        data_source2 = config_source
        data_source1 = alternative_source
        try:
            data_path = proj_dir.parent / data_source1 / f'{var_data_csv}.csv'
            var_data = pd.read_csv(data_path)
        except:
            data_path = proj_dir / data_source2 / f'{var_data_csv}.csv'
            var_data = pd.read_csv(data_path)
    else:
        data_source1 = config_source
        data_source2 = alternative_source
        try:
            data_path = proj_dir / data_source1 / f'{var_data_csv}.csv'
            var_data = pd.read_csv(data_path)
        except:
            data_path = proj_dir.parent / data_source2 / f'{var_data_csv}.csv'
            var_data = pd.read_csv(data_path)

    if var_data is None:
        print(f'Error reading {data_type} data for {var_data_csv} from {data_source1}, {data_source2}', file=sys.stderr, flush=True)
        return None, None

    var_data = remove_extra_index(var_data)
    return data_path, var_data


def pull_raw_data(config, proj_dir, var_ids, alias=True, data_source='data'):
    """
    Pulls raw data for specified variable IDs from the configuration and merges them into a single DataFrame.
    Args:
        config (Config): The configuration object containing data source information.
        proj_dir (Path): The project directory path.
        var_ids (List[str]): List of variable IDs to pull data for.
        alias (bool): Whether to use variable aliases from the config. Default is True.
        data_source (str): The data source to use ('data', 'raw_data', 'master_data', 'master'). Default is 'data'.
    Returns:
        pd.DataFrame: Merged DataFrame containing time variable and specified variables.

    Notes:
        1. The function retrieves the time variable name and its alias from the config.
        2. For each variable ID, it fetches the corresponding data variable name and its alias (if alias=True).
        3. It attempts to read the data from the specified data source, handling potential errors.
        4. The function checks for the presence of the time variable in the data and adjusts if necessary.
        5. It merges all individual variable DataFrames on the time variable, dropping rows with all NaN values in the specified variables.
        6. Returns the final merged DataFrame.

    Calls:
        choose_data_source(proj_dir, config, data_source, var_data_csv)
        check_time_var(var_data, time_var)
    """

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
        data_path, var_data = choose_data_source(proj_dir, config, data_source, var_data_csv=var_data_csv)
        print(f'Using data from {data_path}', file=sys.stdout, flush=True)

        if var_data is None:
            print(f'Error reading raw data for {var_id} from {data_path}', file=sys.stderr, flush=True)
            continue

        if time_var not in var_data.columns:
            time_var = check_time_var(var_data, time_var)

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
    """
    Concatenates a list of DataFrames if not empty, otherwise returns an empty DataFrame.
    Args:
        lst (List[pd.DataFrame]): List of DataFrames to concatenate.

    Returns:
        pd.DataFrame: Concatenated DataFrame or empty DataFrame.
    """
    if len(lst) > 0:
        return pd.concat(lst)
    else:
        return pd.DataFrame()