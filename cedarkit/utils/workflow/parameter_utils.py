import csv
import itertools
import os
import re
import sys
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

import numpy as np
import pandas as pd

try:
    from cedarkit.utils.routing.paths import set_grp_path, template_replace
except ImportError:
    from utils.routing.paths import set_grp_path, template_replace


def remove_already_completed(pset, output_location, performance_consideration=False, existence_consideration=False,
                             max_libsize=375, config=None, source= 'parquet'):
    '''
    Check if a given parameter set has already been computed by looking for existing output files.
    :param pset:
    :param output_location:
    :param performance_consideration:
    :param existence_consideration:
    :param max_libsize:
    :param config:
    :return: [pset] if not computed, else None

    '''
    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']

    param_suffix = f'{pset["surr_var"]}{pset["surr_num"]}'

    knn = pset['knn']
    try:
        if source == 'csv':
            level = 'dir_structure_csv'
            grp_path = set_grp_path(output_location, pset, config=config, make_grp=False, source=source)
            if grp_path.exists() is False:
                print('\tGrp does not exist', grp_path)
                return [pset]
            calc_dir_list = os.listdir(grp_path)
            # print('calc_dir_list', calc_dir_list)
            if len(calc_dir_list) == 0:
                # print('\tGrp empty', grp_path)
                return [pset]

            target_files = [file for file in calc_dir_list if param_suffix in file]
            # print('target_files', target_files)
            if len(target_files) > 0:
                return None
            else:
                return [pset]

        else:
            level = 'dir_structure'
            # print(level, output_location, source)
            grp_path = set_grp_path(output_location, pset, config=config, make_grp=False, source=source)
            parquet_file = template_replace("E{E}_tau{tau}_lag{lag}.parquet", pset, return_replaced=False)
            parquet_file_path = grp_path / parquet_file
            if parquet_file_path.exists() is True:
                dset = ds.dataset(str(grp_path / parquet_file), format="parquet")

                try:
                    unique_val_d = {col: pc.unique(dset.to_table(columns=[col]).column(col)).to_pylist() for col, value in pset.items() if
                       (value is not None) and (col in dset.schema.names)}
                    # print(unique_val_d, pset)
                    existance_d = {}
                    for col in unique_val_d.keys():
                        # print(unique_val_d[col], pset[col])
                        if pset[col] in unique_val_d[col]:
                            existance_d[col]= True
                        else:
                            existance_d[col]=False

                    false_vals = [val for key, val in existance_d.items() if val is False]

                    if len(false_vals) > 0:
                        # print('False', existance_d)
                        return [pset]
                    else:
                        # print('all good', existance_d)
                        return None
                except Exception as e:
                    print('unique value error', e)
                         #     #     #              len(pc.unique(real_table[col]).to_pylist()) > 1}
                # dictionary of filter requests

                # pset = {key: correct_iterable(value) for key, value in pset.items() if key not in ['pset_id', 'id']}
                # filters = {key: ds.field(key).isin(value) for key, value in pset.items() if
                #            value is not None and key in dset.schema.names}
                # # print(filters)
                #
                # # combine filters (AND all of them):
                # combined_filter = reduce(operator.and_, filters.values())
                # table = dset.to_table(filter=combined_filter)
                # # print(table.schema, file=sys.stderr)
                # # df = table.to_pandas()  # already column-subsetted
                # # print(f"Parquet scan found {df.shape} rows after filtering for {pset}", file=sys.stderr)
                # if table.num_rows > 0:
                #     return None
                # else:
                #     return [pset]
            else:
                return [pset]
        # print('\tchecking grp at', grp_path)
    except:
        print('could not find group', pset)

    # if grp_path.exists() is False:
    #     print('\tGrp does not exist', grp_path)
    #     return  [pset]
    # print(1)
    # calc_dir_list = os.listdir(grp_path)
    # # print('calc_dir_list', calc_dir_list)
    # if len(calc_dir_list) == 0:
    #     # print('\tGrp empty', grp_path)
    #     return [pset]

    # knn = pset['knn']

    # conditions for running
    # if source== 'csv':
    #     # print(2, 'csv')
    #     target_files = [file for file in calc_dir_list if param_suffix in file]
    #     # print('target_files', target_files)
    #     if len(target_files) > 0:
    #         return None
    #     else:
    #         return [pset]

    # elif source == 'parquet':
    #
    #     # E, tau = _find_e_tau_dir(grp_path)
    #     # # print('reading parquet for', grp_d, 'at', grp_path, file=sys.stdout)
    #     # parquet_path = _results_parquet_path(grp_path)
    #     # # print(f"Reading Parquet data from {parquet_path} for E={E}, tau={tau}, knn={knn}", file=sys.stdout)
    #     # out = {"path": str(parquet_path), "exists": parquet_path.exists(), "is_file": parquet_path.is_file(),
    #     #        "is_dir": parquet_path.is_dir()}
    #     # # print(out, file=sys.stderr)
    #     #
    #     # # print('attempting to open parquet at', parquet_path/'results.parquet', file=sys.stderr)
    #     # dset = ds.dataset(str(parquet_path / 'results.parquet'), format="parquet")
    #     # # print('opened parquet dataset', file=sys.stderr)
    #     # # dictionary of filter requests
    #     # filters = {key: ds.field(key).isin(value) for key, value in grp_d.items() if
    #     #            value is not None and key in dset.schema.names}
    #     # # print('filters', filters, grp_d)
    #     # # combine filters (AND all of them):
    #     # combined_filter = reduce(operator.and_, filters.values())
    #     # table = dset.to_table(filter=combined_filter)
    #     # print(f"Parquet scan found {table.num_rows} rows after initial filtering", file=sys.stderr)
    #     #
    #
    #     #@TODO need to fix the file name format
    #     # if os.path.exists(grp_path / 'results.parquet') is False:
    #     #     print('\tNo results.parquet found', grp_path / 'results.parquet')
    #     #     return [pset]
    #     # dset = ds.dataset(str(grp_path / 'results.parquet'), format="parquet")
    #     #
    #     print(len(calc_dir_list), 'files in grp dir', grp_path)
    #     for file in calc_dir_list:
    #         dset = ds.dataset(str(grp_path / file), format="parquet")
    #         # dictionary of filter requests
    #         pset = {key: correct_iterable(value) for key, value in pset.items()}
    #         filters = {key: ds.field(key).isin(value) for key, value in pset.items() if value is not None and key in dset.schema.names}
    #
    #         # combine filters (AND all of them):
    #         combined_filter = reduce(operator.and_, filters.values())
    #         table = dset.to_table(filter=combined_filter)
    #         # print(table.schema, file=sys.stderr)
    #         # df = table.to_pandas()  # already column-subsetted
    #         # print(f"Parquet scan found {df.shape} rows after filtering for {pset}", file=sys.stderr)
    #         if table.num_rows > 0:
    #             return None
    #         else:
    #             return [pset]




    # real_dfs_references = [file for file in calc_dir_list if 'neither' in file]
    # if len(real_dfs_references) > 0:
    #     real_df_full = get_real_output(real_dfs_references, grp_path, meta_variables, max_libsize, knn)
    #     if real_df_full is None:
    #         torun.append(pset)
    #     else:
    #         return None
    # else:
    #     # print('\tNo real data found; skipping', grp_path)
    #     torun.append(pset)
    #     # continue
    # #     real_df_full.drop(columns=['Unnamed: 0'], inplace=True)
    # #     ct = 0
    # #     for rel, grp in real_df_full.groupby('relation'):
    # #         min_libsize_rho = grp[grp['LibSize']<50]['rho'].min()
    # #         max_libsize_rho = grp[(grp['LibSize']> 200) &(grp['LibSize']<250)]['rho'].max()
    # #         if max_libsize_rho - min_libsize_rho < 0.1:
    # #             ct += 1
    # #
    # #     if ct == 2:
    # #         print('skipping', grp_path)
    # #         performance_continue = False
    # #         continue
    # #
    # # existence_continue = True
    # # stem = f"E{pset['E']}_tau{pset['tau']}__{pset['surr_var']}{pset['surr_num']}.csv"
    # # written_files = [file for file in os.listdir(grp_path) if stem in file]
    # # if len(written_files) > 0:
    # #     existence_continue = False
    # #     continue
    # # else:
    # #     torun.append(pset)
    # # _parameter_df = pd.DataFrame(torun)
    #
    # # print(f'number of parameter combinations to run: {len(_parameter_df)}, {len(_parameter_df.groupby(["E", "tau", "Tp", "lag","col_var_id", "surr_var"]))} unique combinations')
    # return torun#_parameter_df
    #


def get_assessed_param_picks(proj_dir, output_location, comb_df,config=None,parameter_dir=None,
                             surr=False, surr_num=201, groupby_vars = None, testmode=True,
                           tp_vals = [1], knn_vals = [20],  append=False,source=None,
                             surr_vars=None, verbose= False):
    """
    Generate slurm scripts for running CCM parameters based on combinations in comb_df.

    :param proj_dir: Path to the project directory.
    :param output_location: Path to the output directory where results will be saved.
    :param comb_df: DataFrame containing combinations of parameters to run.
    :param config: Configuration object. If None, it will be loaded from proj_dir/proj_config.yaml.
    :param parameter_dir: Directory where parameter files are stored. If None, defaults to 'parameters' in proj_dir.
    :param surr: Boolean indicating whether to include surrogates in the parameters.
    :param surr_num: Number of surrogate variables to consider.
    :param groupby_var: List of variables to group by when generating parameters.
    :param testmode: Boolean indicating whether to run in test mode (no actual generation).
    :param tp_vals: List of values for the Tp parameter.
    :param knn_vals: List of values for the knn parameter.
    :param suffix: Suffix to append to the generated parameter files.
    :param append: Boolean indicating whether to append to existing files.
    :param proj_prefix: Prefix to use for the project name in the generated files.
    :param default_calc_length: Default calculation length for the CCM.
    :param ntasks: Number of tasks to run in parallel.
    :param max_time_ask: Maximum time to ask for in the slurm script.
    :param verbose: Boolean indicating whether to print verbose output.
    :return: if testmode is True, returns the number of unique parameter combinations; otherwise, generates slurm scripts and prints prompt.

    Uses:
        remove_already_completed(pset, output_location, performance_consideration=False,
                                                  existence_consideration=False,source=source, max_libsize=375, config=config)


    """

    if surr is True:
        surr_num_max = surr_num
        surr_num_min = 1
        if surr_vars is None:
            if config is not None:
                surr_vars = [config.col.var, config.target.var]#surr_vars

            if 'surr_var' in comb_df.columns:
                if len(comb_df.surr_var.unique()) == 1:
                    if comb_df.surr_var.unique()[0] != 'neither':
                        surr_vars = comb_df.surr_var.unique()


            if surr_vars is None:
                print('config missing, please provide to infer surrogate variables. Alternatively, state surrogate variables')
                return []
    else:
        surr_num_max = 1
        surr_vars = ['neither']
        surr_num_min = 0

    messages = []
    messages.append(f'Generating parameters for slurm scripts: Testmode: {testmode}')
    messages.append(f'proj_dir: {proj_dir}')
    messages.append(f'output location: {output_location}')
    messages.append(f'\tcomb_df: {comb_df.shape}, surr bool: {surr}, target surr var/num: {surr_vars}; {surr_num}, groupby_var: {groupby_vars}')

    if testmode is True:
        messages.append('\t! --> testmode is True, not generating anything')
    else:
        messages.append(f'Slurm scripts will be generated in:, {proj_dir} / slurm')
    messages.append('------------------------------------------------------------')


    if parameter_dir is None:
        parameter_dir = proj_dir / 'parameters'
    if groupby_vars is None:
        groupby_vars = ['E', 'tau', 'col_var_id','target_var_id']  # (E, tau, col_var_id, target_var_id)#['E', 'tau']#['E', 'tau']

    if len(surr_vars) > 1:
        groupby_vars.append('surr_var')

    # print(comb_df.head())
    parameter_subs = []
    for col in ['E', 'tau', 'lag', 'knn', 'surr_num', 'Tp']:
        if col in comb_df.columns:
            comb_df[col] = comb_df[col].astype(int)


    for col_var, comb_grp_df in comb_df.groupby('col_var_id'):
        parameter_df_master = pd.read_csv(parameter_dir / f'params_bycol_{col_var}.csv')
        for col in ['E', 'tau', 'lag', 'knn', 'surr_num', 'Tp']:
            parameter_df_master[col] = parameter_df_master[col].astype(int)
        # this happened once that surr_var = neither but surr_num > 0
        parameter_df_master = parameter_df_master[~((parameter_df_master['surr_var'] == 'neither') & (parameter_df_master['surr_num'] != 0))].copy()
        # print(parameter_df_master.head())

        for (target_var, E_val, tau_val, lag_val), subcomb_grp_df in comb_grp_df.groupby(['target_var_id', 'E', 'tau', 'lag']):

            target_vars = [target_var]
            E_vals = [E_val]
            tau_vals = [tau_val]
            lag_vals = [lag_val]
                        # for surr_var in comb_grp_df.surr_var.unique():
                        #     surr_vars = [surr_var]
            conditions = np.all([
                                (parameter_df_master['E'].isin( E_vals)),
                                (parameter_df_master['tau'].isin( tau_vals)),
                                (parameter_df_master['lag'].isin( lag_vals)),
                                (parameter_df_master['target_var_id'].isin( target_vars)),
                                (parameter_df_master['surr_var'].isin( surr_vars)),
                                (parameter_df_master['surr_num'] < surr_num_max), (parameter_df_master['surr_num'] >= surr_num_min),
                                (parameter_df_master['knn'].isin(knn_vals)),
                                (parameter_df_master['Tp'].isin(tp_vals)),
                            ], axis=0)
            # print((target_vars, E_vals, tau_vals, lag_vals), surr_vars, surr_num_max, knn_vals, tp_vals)
            parameter_sub_df = parameter_df_master[conditions].copy()
            # print(parameter_sub_df.head())
            parameter_subs.append(parameter_sub_df)

    parameter_df = pd.concat(parameter_subs)
    # print('head', parameter_df.head())

    # for col in parameter_df.columns:
    #     parameter_df[col] = parameter_df[col].apply(lambda x: x if ~isinstance(x, list) else x[0] if (len(x)==1) else x)
    for col in ['E', 'tau', 'Tp', 'knn', 'surr_num', 'sample', 'lag', 'train_ind_i', 'train_len']:
        try:
            parameter_df[col] = parameter_df[col].astype(int)
        except:
            print('1 could not convert', col)
            pass
    parameter_ds = parameter_df.to_dict(orient='records')
    done = []
    torun = []
    for ik, pset in enumerate(parameter_ds):
        if append is not True:
            candidates = remove_already_completed(pset, output_location, performance_consideration=False,
                                                  existence_consideration=False, source=source, max_libsize=375, config=config)
            # print('checking', ik, 'of', len(parameter_ds), pset, 'candidates', candidates)
            if candidates is None:
                done.append(pset)
            else:
                torun += candidates
        else:
            # print('append', pset)
            torun.append(pset)

    to_run_df = pd.DataFrame.from_records(torun)
    to_run_df['to_run'] = True
    done_df = pd.DataFrame.from_records(done)
    done_df['to_run'] = False

    combined_df = pd.concat([to_run_df, done_df], ignore_index=True)
    # for col in combined_df.columns:
    #     combined_df[col] = combined_df[col].apply(lambda x: x if ~isinstance(x, list) else x[0] if (len(x)==1) else x)
    combined_df = combined_df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    # print(combined_df.head())
    for col in ['E', 'tau', 'lag', 'knn', 'surr_num', 'Tp', 'sample', 'id']:
        try:
            combined_df[col] = combined_df[col].astype(int)
        except:
            print('could not convert', col)

    return combined_df, messages


parameters_d = {
    'tau': {
        'values': np.arange(1, 7, 1)  # A range of tau values from 4 to 12
    },
    'E': {
        'values': np.arange(3, 11, 1)  # A range of embedding dimensions (E) from 4 to 15
    },
    'train_len': {
        'values': [None]  # Train length value, set to None for now
    },
    'train_ind_i': {
        'values': [0]  # Train index value, set to 0 by default
    },
    'knn': {
        'values': [7, 10]  # Number of nearest neighbors (knn) fixed at 20
    },
    'Tp_flag': {
        'values': [None]  # Placeholder for Tp flag values
    },
    'Tp': {
        'values': [2, 4]  # Prediction horizon (Tp) fixed at 20
    },
    'lag': {
        'values': [-8, -6, -4,-2, 0,2,  4,6,  8]# np.arange(-38, 39, 4)  # A range of lag values from -38 to 38 in steps of 4
    },
    'Tp_lag_total': {
        'values': [32]  # Total lag value fixed at 32
    },
    'sample': {
        'values': [200]#[100]  # Sample size fixed at 100
    },
    'weighted': {
        'values': [False]  # Whether to use weighted calculation, set to False
    },
    'target_var': {
        'values': []  # Placeholder for target variable values
    },
    'col_var': {
        'values': []  # Placeholder for column variable values
    },
    'surr_var': {
        'values': ['neither']  # Surrogate variable, default is 'neither'
    },
    'surr_num': {
        'values': [1]#np.arange(1, 11, 1)  # Range of surrogate numbers from 1 to 19
    },
}


def process_params_group(arg_tuple, write_mode='a'):
    '''
    Process a group of arguments to generate parameter combinations and write them to a CSV file.
    Args:
        arg_tuple (tuple): A tuple containing various arguments needed for processing.
        write_mode (str): The mode to open the CSV file ('a' for append, 'w' for write).


    '''
    (col_var_id, col_var_alias, target_var_id, target_var_alias,
     surrogate_vars,surrogate_range,
     param_flags, param_csv, parameters) = arg_tuple

    # Get the names of all parameters
    param_names = list(parameters.keys())

    # Assign target and column variable aliases from the provided arguments
    parameters['target_var']['values'] = [target_var_alias]
    parameters['col_var']['values'] = [col_var_alias]

    if len(surrogate_vars)==0:
        surrogate_vars = ['neither']
    parameters['surr_var']['values'] = surrogate_vars

    # If additional surrogate instructions are provided, set the range for surrogate numbers
    if len(surrogate_range) > 1:
        parameters['surr_num']['values'] = np.arange(surrogate_range[0], surrogate_range[1], 1)

    # Pattern for identifying tau multiples (from param flags)
    pattern = r'\d+'
    tau_multiples = []

    # If flags include 'Tp_tau', extract the tau multiple values
    if len(param_flags) > 0:
        tau_multiples = [int(re.findall(pattern, flag)[0]) for flag in param_flags if 'Tp_tau' in flag]

    # List to store all generated parameter sets
    parameter_sets = []

    # If tau multiples are specified, generate combinations with tau multiples
    if len(tau_multiples) > 0:
        for tau_multiple in tau_multiples:
            for tau in parameters['tau']['values']:
                copy_params = parameters.copy()  # Make a copy of the parameters to modify
                copy_params['Tp']['values'] = [tau * tau_multiple]  # Set Tp based on tau_multiple
                copy_params['tau']['values'] = [tau]  # Set tau value
                param_values = [copy_params[param]['values'] for param in param_names]  # Extract param values
                parameter_sets.append(param_values)  # Append to parameter sets

    # Otherwise, generate parameter combinations without tau multiples
    else:
        param_values = [parameters[param]['values'] for param in param_names]
        parameter_sets.append(param_values)  # Use itertools.product for all combinations
    # Determine if the CSV file already exists and set write mode
    skip_header = True

    if param_csv.exists()==False:  # If the CSV file does not exist, create a new one
        write_mode = 'w'
        skip_header = False

    # Write the parameter combinations to the CSV file
    with open(param_csv, write_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        header = ['id'] + param_names + ['col_var_id', 'target_var_id']
        if not skip_header:
            writer.writerow(header)

        # Write each parameter combination along with a unique ID
        for param_values in parameter_sets:
            for values in itertools.product(*param_values):
                unique_id = int(time.time() * 1000)
                new_row = [unique_id] + list(values) + [col_var_id, target_var_id]# Create a unique ID based on the current time
                writer.writerow(new_row)
                time.sleep(0.001)  # Sleep for a millisecond to ensure unique IDs

    # print(f"CSV file {param_csv} has been created.")
    # param_df = pd.read_csv(param_csv)
    # print(f"{param_csv} has been read and has length: {len(param_df)}", file=sys.stdout, flush=True)
    # param_df = param_df.drop_duplicates(subset=param_df.columns.difference(['id']), keep='first')
    # print(f"{param_csv} duplicates have been dropped and now has length: {len(param_df)}", file=sys.stdout, flush=True)

    # param_df.to_csv(param_csv, index=False)


def tidy_up_params(param_csv_paths, keep='first'):
    param_csv_paths = list(set(param_csv_paths))
    for param_csv_path in param_csv_paths:

        param_df = pd.read_csv(param_csv_path)
        print(f"{param_csv_path} has been read and has length: {len(param_df)}", file=sys.stdout, flush=True)
        param_df = param_df.drop_duplicates(subset=param_df.columns.difference(['id']), keep=keep)
        print(f"{param_csv_path} duplicates have been dropped and now has length: {len(param_df)}", file=sys.stdout,
              flush=True)
        param_df_summary = (
            param_df.groupby(['col_var_id', 'target_var_id', 'E', 'tau', 'lag', 'surr_var'])
            .agg(min_surr_num=('surr_num', 'min'), max_surr_num=('surr_num', 'max'))
            .reset_index()
        )
        param_df_summary.sort_values(by=['col_var_id', 'target_var_id', 'max_surr_num', 'E', 'tau', 'lag', 'surr_var'], inplace=True)

        if isinstance(param_csv_path, str):
            param_csv_path = Path(param_csv_path)
        summary_csv_path = param_csv_path.parent / f'summary_{param_csv_path.name}'
        param_df_summary.to_csv(summary_csv_path, index=False)
        print(f"Summary CSV file {summary_csv_path} has been created.", file=sys.stdout, flush=True)

        param_df.to_csv(param_csv_path, index=False)


def make_comb_df(col_var_ids=None, target_var_ids=None, E_tau_combs=None, lag_vals=None, tp_vals=None, knn_vals=None, df_path=None):
    if df_path is not None:
        comb_df = pd.read_csv(df_path)
        return comb_df
    else:
        assert col_var_ids is not None
        assert target_var_ids is not None
        assert E_tau_combs is not None
        assert lag_vals is not None
        assert tp_vals is not None
        assert knn_vals is not None
    comb_list = []
    for col_var_id in col_var_ids:
        for target_var_id in target_var_ids:
            for pair in E_tau_combs:
                E, tau = pair
                for lag in lag_vals:
                    for tp in tp_vals:
                        for knn in knn_vals:
                            comb_list.append({
                                'E': E,
                                'tau': tau,
                                'lag': lag,
                                'Tp': tp,
                                'knn': knn,
                                'col_var_id': col_var_id,
                                'target_var_id': target_var_id,
                            })
    comb_df = pd.DataFrame(comb_list)
    return comb_df
