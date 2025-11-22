from pathlib import Path
import pandas as pd
import os
try:
    from cedarkit.utils.routing.paths import check_location
except ImportError:
    # Fallback: imports when running as a package
    from utils.routing.file_name_parsers import remove_extra_index
    from utils.routing.paths import check_location


#
# def correct_iterable(obj):
#     if obj is None:
#         return None
#     if isinstance(obj, str):
#         return [obj]
#     else:
#         if isinstance(obj, collections.abc.Iterable):
#             return obj
#         else:
#             return [obj]


#

# def pull_data(col_var_id, source_units, var_name, var_generic, data_var, modifier, source_df=None, data_csv=None):
#     master_dir = 'master_data'
#     # if data_csv is None:
#     #     data_csv = f'{col_var_id}_{var}_{modifier}'.strip('_')
#     if source_df is None:
#         source_df = pd.read_csv(proj_dir.parent / master_dir / f'{data_csv}.csv', index_col=0)
#
#     time_var = 'time' if 'time' in source_df.columns else 'date'
#     if data_var not in source_df.columns:
#         data_var = f'{data_var}_{modifier.replace("detrended", "")}'.strip(
#             '_')  # if 'detrended' in modifier else data_var
#     # print(source_df.columns, data_var)
#     source_ps = pyleo.Series(time=-source_df[time_var].values, value=source_df[data_var].values,
#                              time_unit='yr BP', value_unit=source_units, value_name=var_name,
#                              # label='wu_tsi')
#                              label=data_var)
#
#     # source_ps.label = f'{col_var_id}_{var}_{modifier}'
#
#     return source_df, source_ps


# def pull_surrogates(col_var_id, source_units, var_name, var, data_var, modifier, surr_csv=None):
#     master_dir = 'master_surrogates'
#     if surr_csv is None:
#         prefix = 'pyleo_surr_'
#         surr_csv = f'{prefix}{col_var_id}_{var}_{modifier}'.strip('_')
#     surr_csv = surr_csv.replace('.csv', '')  # remove .csv if it exists
#     surr_df = pd.read_csv(proj_dir.parent / master_dir / f'{surr_csv}.csv', index_col=0)
#
#     surr_time = -surr_df['date'].values
#     surr_values_df = surr_df[surr_df.columns[surr_df.columns != 'date']]
#
#     surr_ms = pyleo.MultipleSeries([pyleo.Series(time=surr_time, value=surr_values_df[surr_col].values,
#                                                  time_unit='yr BP', value_unit=source_units, value_name=var_name,
#                                                  # label='wu_tsi')
#                                                  label=surr_col) for surr_col in surr_values_df.columns])
#     return surr_df, surr_ms


# def package_masters(proj_dir, detrended_ps, detrended_surr_ms, col_var_id, data_var, var, modifier,save_choices=[], replace=False, mode='test', file_name=None):
def package_masters(proj_dir, var_obj, detrended_surr_ms, col_var_id=None, data_var=None, var_generic=None, modifier=None, save_choices=[],
                        replace=False, mode='test', file_name=None, priority='local'):

    """
    Save detrended data and surrogates to master_data and master_surrogates folders, respectively.

    Parameters
    ----------
    proj_dir : Path
        Project directory path.
    detrended_ps : pyleo.Series
        Detrended time series data.
    detrended_surr_ms : pyleo.MultipleSeries
        Detrended surrogate time series data.
    col_var_id : str
        Identifier for the column variable.
    data_var : str
        Name of the data variable.
    var : str
        Variable name.
    modifier : str
        Modifier for the variable (e.g., 'detrended').
    save_choices : list, optional
        List of choices to save ('orig', 'surr', or both). Default is [].
    replace : bool, optional
        Whether to replace existing files. Default is False.
    mode : str, optional
        Mode of operation ('test' or 'save'). Default is 'test'.
    file_name : str, optional
        Custom file name for saving. If None, a default name is generated. Default is None.

    Returns
    -------
    data_df : pd.DataFrame
        DataFrame containing the detrended data.


    Notes
    -----
    The function checks for existing files and saves the detrended data and surrogates
    to the specified directories based on the provided options.
    1. If `save_choices` is empty, the function prints a message and returns the data DataFrame without saving.
    2. If `file_name` is not provided, it generates a default file name based on `col_var_id`, `var`, and `modifier`.
    3. The function saves the detrended data to both `master_data` and `data` directories if they exist.
    4. The function saves the surrogate data to both `master_surrogates` and `surrogates` directories if they exist.
    5. The function handles file existence checks and respects the `replace` and `mode` parameters.

    Examples
    --------
    >>> data_df = package_masters(proj_dir, detrended_ps, detrended_surr_ms, 'var_id', 'data_var', 'var', 'detrended', save_choices=['orig', 'surr'], replace=True, mode='save')
    >>> print(data_df.head())
    >>> data_df = package_masters(proj_dir, detrended_ps, detrended_surr_ms, 'var_id', 'data_var', 'var', 'detrended', save_choices=['orig'], mode='test')
    >>> print(data_df.head())
    >>> data_df = package_masters(proj_dir, detrended_ps, detrended_surr_ms, 'var_id', 'data_var', 'var', 'detrended', save_choices=['surr'], replace=False, mode='save', file_name='custom_name')
    >>> print(data_df.head())

    """

    return_dfs = {'data': None, 'surr': None}
    real_time_var = 'time'  #'date' #'time'  #config.raw_data.time_var
    surr_time_var = 'time'  #'date' #'time'  #config.surrogate_data.time_var
    if var_obj is not None:
        col_var_id = var_obj.var_id
        var_generic = var_obj.var
        data_var = var_obj.col_name
        detrended_ps = var_obj.ps
        real_time_var = var_obj.real_ts_time
        surr_time_var = var_obj.surr_ts_time

        if var_obj.real_csv_stem is not None:
            file_name = var_obj.real_csv_stem
        if (modifier is not None) and (modifier !=''):
            if modifier in file_name:
                if '_'+modifier not in file_name:
                    file_name = file_name.replace(modifier, '_' + modifier)
            else:
                file_name = f'{file_name}_{modifier}'
    else:
        if file_name is None:
            file_name = f'{col_var_id}_{var_generic}_{modifier}'.strip('_')

    # file_name = f'{col_var_id}_{var}_{modifier}'.strip('_')
    data_d = {'time': -detrended_ps.time, f'{data_var}'.strip('_'): detrended_ps.value}
    data_df = pd.DataFrame(data_d).sort_values(by='time').reset_index(drop=True)
    data_df.rename(columns={'time': real_time_var}, inplace=True)
    return_dfs['data'] = data_df

    if len(save_choices) == 0:
        print('no save choices provided, not saving anything. Choose from "orig", "surr" or both')
        return return_dfs
    if isinstance(save_choices, str):
        save_choices = [save_choices]

    existing_files  = [f for f in os.listdir(proj_dir.parent / 'master_data' ) if file_name in f]
    if 'orig' in save_choices:
        if replace is False:
            if len(existing_files) >0:
                suffix_nums = [int(f.replace(file_name,'').replace('.csv','').replace('_','')) for f in existing_files if f.replace(file_name,'').replace('.csv','').replace('_','').isdigit()]
                if len(suffix_nums) ==0:
                    suffix_nums = [0]
                if len(suffix_nums) >0:
                    max_suffix = max(suffix_nums)+1
                    file_name = f'{file_name}_{max_suffix}'

        exists = os.path.exists(proj_dir.parent / 'master_data' / f'{file_name}.csv')
        if exists is False:
            save_paths = []
            if priority == 'local':
                save_paths.append(proj_dir / 'data' / f'{file_name}.csv')
            elif priority == 'master':
                save_paths.append(proj_dir.parent / 'master_data' / f'{file_name}.csv')
            else:
                save_paths.append(proj_dir.parent / 'master_data' / f'{file_name}.csv')
                save_paths.append(proj_dir / 'data' / f'{file_name}.csv')

            if mode == 'test':
                print(f'File would be saved for scope: {priority} to:', save_paths)

            elif mode == 'save':
                for save_path in save_paths:
                    print(f'Saving to {save_path}')
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    data_df.to_csv(save_path)
                # print(f'Saving {proj_dir.parent}/master_data / {file_name}.csv')
                # data_df.to_csv(proj_dir.parent / 'master_data' / f'{file_name}.csv')
        else:
            print(f'File {proj_dir.parent}/master_data / {file_name}.csv already exists, skipping saving.')

        # if os.path.exists(proj_dir / 'data') is True:
        #     exists = os.path.exists(proj_dir / 'data' / f'{file_name}.csv')
        #     if replace is True:
        #         exists = False
        #
        #     if exists is False:
        #         if mode == 'test':
        #             print(f'Would save to {proj_dir}/data / {file_name}.csv')
        #         elif mode == 'save':
        #             print(f'Saving {proj_dir}/data / {file_name}.csv')
        #             data_df.to_csv(proj_dir / 'data' / f'{file_name}.csv')
        #     else:
        #         print(f'File {proj_dir}/data / {file_name}.csv already exists, skipping saving.')

    if 'surr' in save_choices:
        detrended_surr_file_name = f'pyleo_surr_{file_name}'.strip('_')
        detrended_surr_d = {
            'time': -detrended_ps.time}  # , 'value': orig.value, 'source': orig.label, 'var': orig.value_name}

        for ik in range(len(detrended_surr_ms.series_list)):
            detrended_surr_d[f'{var_generic}_{ik + 1}'] = detrended_surr_ms.series_list[ik].value

        # if mode == 'save':
        detrended_surr_df = pd.DataFrame(detrended_surr_d)
        detrended_surr_df.sort_values(by='time', inplace=True)
        detrended_surr_df.reset_index(drop=True, inplace=True)
        detrended_surr_df.rename(columns={'time': surr_time_var}, inplace=True)
        return_dfs['surr'] = detrended_surr_df

        exists = os.path.exists(proj_dir.parent / 'master_surrogates' / f'{detrended_surr_file_name}.csv')
        if replace is True:
            exists = False

        if exists is False:
            if mode == 'test':
                print(f'Would save to {proj_dir.parent}/master_surrogates / {detrended_surr_file_name}.csv')
            elif mode == 'save':
                print(f'Saving {proj_dir.parent}/master_surrogates / {detrended_surr_file_name}.csv')
                detrended_surr_df.to_csv(proj_dir.parent / 'master_surrogates' / f'{detrended_surr_file_name}.csv')
        else:
            print(
                f'File {proj_dir.parent}/master_surrogates / {detrended_surr_file_name}.csv already exists, skipping saving.')

        # (proj_dir / 'surrogates').mkdir(parents=True, exist_ok=True)
        if os.path.exists(proj_dir / 'surrogates') is True:
            exists = os.path.exists(proj_dir / 'surrogates' / f'{detrended_surr_file_name}.csv')
            if replace is True:
                exists = False
            if exists is False:
                if mode == 'test':
                    print(f'Would save to {proj_dir}/surrogates / {detrended_surr_file_name}.csv')
                elif mode == 'save':
                    print(f'Saving {proj_dir}/surrogates / {detrended_surr_file_name}.csv')
                    detrended_surr_df.to_csv(proj_dir / 'surrogates' / f'{detrended_surr_file_name}.csv')
            else:
                print(f'File {proj_dir}/surrogates / {detrended_surr_file_name}.csv already exists, skipping saving.')
        else:
            print(f'No surrogates directory found at {proj_dir}/surrogates, skipping saving there.')
            # return data_df

        # if mode == 'test':
        #     print(f'Would save to {proj_dir.parent}/master_surrogates / {detrended_surr_file_name}.csv')
        # elif mode== 'save':
        #     print(f'Saving {proj_dir.parent}/master_surrogates / {detrended_surr_file_name}.csv')
        #     detrended_surr_df.to_csv(proj_dir.parent/'master_surrogates' / f'{detrended_surr_file_name}.csv')#, index=False)

    return return_dfs

# def make_slurm_script(E_grp, new_param_file, new_file_name, slurm_dir, source_file_path, default_calc_length=25,
#                       max_time_ask=240, buffer_percent=1.5, ntasks=36, append=False):
#     new_file_path = os.path.join(slurm_dir, new_file_name)
#
#     proj_name = str(slurm_dir.parent.name)
#     proj_dir_name = str(slurm_dir.parent.parent.name)
#     # Copy the file
#     shutil.copy(source_file_path, new_file_path)
#
#     # Read and modify the new file
#     with open(new_file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Replace "export PARAMS=" and "SEQ_END="
#     param_length = len(E_grp) + 1
#     new_lines = []
#     for line in lines:
#         if line.strip().startswith('export PARAMS='):
#             line = f'export PARAMS="{new_param_file}"\n'
#         elif line.strip().startswith('SEQ_END='):
#             # Calculate the length of the new parameter file
#             line = f'SEQ_END={param_length}\n'
#         elif 'PROJECT=' in line.strip():#.startswith('PROJECT='):
#             # Calculate the length of the new parameter file
#             line = line.split('=')[0]+f'={proj_name}\n'#replace('PROJECT=', f'PROJECT=')
#             # line = f'PROJECT={proj_name}\n'
#         elif 'PROJECT_DIR=' in line.strip():#.startswith('PROJECT='):
#             # Calculate the length of the new parameter file
#             line = line.split('/lplander/')[0]+f'/lplander/{proj_dir_name}"\n'#replace('PROJECT=', f'PROJECT=')
#             # line = f'PROJECT={proj_name}\n'
#         elif line.strip().startswith('#SBATCH --ntasks='):
#             # ntasks = int(line.replace('#SBATCH --ntasks=', '').split(' ')[0])
#             ntasks = min(ntasks, param_length)
#             line = f'#SBATCH --ntasks={ntasks}\n'
#         elif append is True:
#             if '$OUTPUT_DIR --cpus' in line:
#                 line = line.replace('$OUTPUT_DIR --cpus', '$OUTPUT_DIR --override --write append --cpus')
#         new_lines.append(line)
#     time_est = int(default_calc_length * param_length / ntasks)
#     time_est_padded = int(min(max_time_ask, min(int(time_est * buffer_percent), time_est + 30)))
#
#     new_lines2 = []
#     for line in new_lines:
#         if line.strip().startswith('#SBATCH --time='):
#             line = f'#SBATCH --time=00:{time_est_padded}:00\n'
#             new_lines2.append(line)
#         else:
#             new_lines2.append(line)
#
#     # Write the modified content back to the file
#     with open(new_file_path, 'w') as file:
#         file.writelines(new_lines2)
#
#     # print(f'File copied and modified: {new_file_path}, param length:{param_length}')
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

    loc_config = config.get_dynamic_attr('{var}', check_location(proj_dir))
    if data_type == 'real':
        try:
            config_source = loc_config.raw_data
        except:
            config_source = 'data'
        alternative_source = 'master_data'
    elif data_type in ['surr', 'surrogate']:
        try:
            config_source = loc_config.surrogate_data
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


def remove_extra_index(df):
    """
    Removes the 'Unnamed: 0' column from the DataFrame if it exists.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df
