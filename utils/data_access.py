from operator import index

import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import datetime
import gc
import re



def check_csv(output_file_name):
    if '.csv' not in output_file_name:
        output_file_name = f'{output_file_name}.csv'
    return output_file_name

def remove_numbers(input_string):
    # Use regex to remove all digits from the string
    return re.sub(r'\d+', '', input_string)

def extract_numbers(input_string):
    # Use regex to extract all numbers from the string
    return re.findall(r'\d+', input_string)[-1]  # Assuming numbers are before the first '__'


def remove_extra_index(df):
    """
    Removes the 'Unnamed: 0' column from the DataFrame if it exists.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

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


# TODO migrate to DataVar
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


