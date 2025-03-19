import numpy as np
import csv
import itertools
import time
import sys
from pathlib import Path
import os
import re
import yaml
from utils.arg_parser import get_parser  # Importing the argument parser from arg_parser.py
import importlib.util
from utils.config_parser import load_config

# PROJECT=eevw
# PARAMS=real

# python make_params.py --project $PROJECT --parameters $PARAMS --inds 1 25 --vars erb wu surrogate temp tsi neither --flags Tp_tau2 Tp_tau3
# makes parameters for the specified project,
# written to the specified parameter file,
# with surrogates numbered 1 to 25,
# using specified variable_ids (default to all combinations)
# and surrogates for temp and tsi (default to neither),
# and additional flags Tp_tau2 and Tp_tau3



# Define a dictionary of parameters to be used for generating combinations later
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


# Function to process a group of arguments for parameter combinations
def process_group(arg_tuple):
    (col_var_id, col_var_alias, target_var_id, target_var_alias,
     surrogate_vars,surrogate_range,
     param_flags, param_csv, parameters) = arg_tuple

    # Get the names of all parameters
    param_names = list(parameters.keys())
    print(param_names)

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
    print(param_csv)
    # Determine if the CSV file already exists and set write mode
    skip_header = True
    write_mode = 'a'
    if param_csv.exists()==False:  # If the CSV file does not exist, create a new one
        write_mode = 'w'
        skip_header = False
    print('write_mode', write_mode)

    # Write the parameter combinations to the CSV file
    with open(param_csv, write_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        header = ['id'] + param_names + ['col_var_id', 'target_var_id']
        print(header)
        if not skip_header:
            writer.writerow(header)

        # Write each parameter combination along with a unique ID
        for param_values in parameter_sets:
            print(param_values)
            for values in itertools.product(*param_values):
                unique_id = int(time.time() * 1000)
                new_row = [unique_id] + list(values) + [col_var_id, target_var_id]# Create a unique ID based on the current time
                print(len(new_row), len(header))
                writer.writerow(new_row)
                time.sleep(0.001)  # Sleep for a millisecond to ensure unique IDs

    print(f"CSV file {param_csv} has been created.")


if __name__ == '__main__':
    # Create the parser object from the argument parser file
    parser = get_parser()
    args = parser.parse_args()  # Parse the command-line arguments

    # Handle the project name argument, which is required (-j / --project)
    if args.project is not None:
        proj_name = args.project  # Store the project name
    else:
        print('Project name is required', file=sys.stdout, flush=True)
        sys.exit(0)

    # Create the project directory path using Pathlib
    proj_dir = Path(os.getcwd()) / proj_name

    # Load the project configuration file specified by the --config flag (if provided)
    gen_config = 'proj_config'
    if args.config is not None:
        gen_config = args.config

    # with open(proj_dir / f'{gen_config}.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    config = load_config(proj_dir / 'proj_config.yaml')

    # Handle the parameter file argument, which is optional (-p / --parameters)
    if args.parameters is not None:
        parameter_flag = args.parameters  # Store the parameter file name
    else:
        print('Parameter file is required', file=sys.stdout, flush=True)
        sys.exit(0)

    parameters_dir = proj_dir / 'parameters'
    # if
    # (proj_dir / 'parameters').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

    # Load the Python file as a module
    print(config.parameters)
    spec = importlib.util.spec_from_file_location("parameters", parameters_dir / f'{config.parameters.spec_d}.py')
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)

    # Now you can access the parameters dictionary
    parameters_d = params_module.parameters_d


    # Handle the number of surrogates specified by the --inds flag
    surrogate_range = []
    if args.inds is not None:
        surrogate_range = args.inds

    # Handle any additional flags provided by the --flags argument
    param_flags = []
    if args.flags is not None:
        param_flags = args.flags

    # Create the path to the output CSV file where parameter combinations will be stored
    param_csv_path = proj_dir / 'parameters' / f'{parameter_flag}.csv'

    # Prepare variable ID tuples for processing
    var_id_tuples = []
    surrogate_vars = []
    surrogate_instructions_ind = None

    # If variables are provided using --vars, process them
    if args.vars is not None:
        surrogate_instructions_ind = [ik for ik, _ in enumerate(args.vars) if _ == 'surrogate'][0]
        if surrogate_instructions_ind is not None:
            surrogate_instructions = args.vars[surrogate_instructions_ind + 1:]
            surrogate_vars = [var for var in surrogate_instructions if type(var) == str]

    specified_vars = False
    if (args.vars is not None):
        if (args.vars[0] != 'surrogate'):
            specified_vars = True

    if specified_vars ==True:
        col_var_id = args.vars[0]
        target_var_id = args.vars[1]
        col_var_alias = config.col.var
        target_var_alias = config.target.var

        var_id_tuples.append((col_var_id, col_var_alias, target_var_id, target_var_alias,
                              surrogate_vars, surrogate_range,
                              param_flags, param_csv_path, parameters_d))
    else:
        # If no variables are provided, iterate over all column and target variable IDs in the config
        for col_var_id in config.col.ids:
            for target_var_id in config.target.ids:
                col_var_alias = config.col.var
                target_var_alias = config.target.var

                var_id_tuples.append((col_var_id, col_var_alias, target_var_id, target_var_alias,
                                      surrogate_vars,surrogate_range,
                                      param_flags, param_csv_path, parameters_d))

    # Process each group of variables and parameters
    for var_id_tuple in var_id_tuples:
        process_group(var_id_tuple)
