import copy
import os
import shutil

import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.workflow import get_assessed_param_picks
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    from utils.workflow.parameter_utils import get_assessed_param_picks
    from utils.cli.logging import setup_logging, log_line

def make_slurm_script(E_grp, new_param_file, new_file_name, slurm_dir, source_file_path, default_calc_length=25,
                      max_time_ask=240, buffer_percent=1.5, ntasks=36, append=False):
    new_file_path = os.path.join(slurm_dir, new_file_name)

    proj_name = str(slurm_dir.parent.name)
    proj_dir_name = str(slurm_dir.parent.parent.name)
    # Copy the file
    shutil.copy(source_file_path, new_file_path)

    # Read and modify the new file
    with open(new_file_path, 'r') as file:
        lines = file.readlines()

    # Replace "export PARAMS=" and "SEQ_END="
    param_length = len(E_grp) + 1
    new_lines = []
    for line in lines:
        if line.strip().startswith('export PARAMS='):
            line = f'export PARAMS="{new_param_file}"\n'
        elif line.strip().startswith('SEQ_END='):
            # Calculate the length of the new parameter file
            line = f'SEQ_END={param_length}\n'
        elif 'PROJECT=' in line.strip():#.startswith('PROJECT='):
            # Calculate the length of the new parameter file
            line = line.split('=')[0]+f'={proj_name}\n'#replace('PROJECT=', f'PROJECT=')
            # line = f'PROJECT={proj_name}\n'
        elif 'PROJECT_DIR=' in line.strip():#.startswith('PROJECT='):
            # Calculate the length of the new parameter file
            line = line.split('/lplander/')[0]+f'/lplander/{proj_dir_name}"\n'#replace('PROJECT=', f'PROJECT=')
            # line = f'PROJECT={proj_name}\n'
        elif line.strip().startswith('#SBATCH --ntasks='):
            # ntasks = int(line.replace('#SBATCH --ntasks=', '').split(' ')[0])
            ntasks = min(ntasks, param_length)
            line = f'#SBATCH --ntasks={ntasks}\n'
        elif append is True:
            if '$OUTPUT_DIR --cpus' in line:
                line = line.replace('$OUTPUT_DIR --cpus', '$OUTPUT_DIR --override --write append --cpus')
        new_lines.append(line)
    time_est = int(default_calc_length * param_length / ntasks)
    time_est_padded = int(min(max_time_ask, min(int(time_est * buffer_percent), time_est + 30)))

    new_lines2 = []
    for line in new_lines:
        if line.strip().startswith('#SBATCH --time='):
            line = f'#SBATCH --time=00:{time_est_padded}:00\n'
            new_lines2.append(line)
        else:
            new_lines2.append(line)

    # Write the modified content back to the file
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines2)

    # print(f'File copied and modified: {new_file_path}, param length:{param_length}')


def gen_parameters_slurm2(proj_dir, output_location, comb_df, min_num_to_run=8, config=None,parameter_dir=None, surr=False, surr_num=201, groupby_var = None, testmode=True,
                           tp_vals = [1], knn_vals = [20], suffix = '', append=False, proj_prefix= 'eevw', default_calc_length=28,surr_vars=None,sample=150,return_combined=False,
                                          source=None, ntasks=42, max_time_ask=300, verbose= False):
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

    """
    print('gen_parameters_slurm2', groupby_var)
    # calls get_assessed_param_picks internally
    combined_df, messages = get_assessed_param_picks(proj_dir, output_location, comb_df,config=config,parameter_dir=parameter_dir, surr=surr, surr_num=surr_num, groupby_vars = copy.copy(groupby_var), testmode=testmode,
                            tp_vals = tp_vals, knn_vals = knn_vals,append=append,surr_vars=surr_vars, verbose= verbose, source=source)
    # print('combined',combined_df.head())
    # calls make_slurm_script internally
    messages2 = gen_slurm_param_from_params(output_location, proj_dir, combined_df, messages=messages, parameter_flag='params',min_num_to_run=min_num_to_run,
                            testmode=testmode, suffix=suffix, proj_prefix=proj_prefix, default_calc_length=default_calc_length,
                            ntasks=ntasks, max_time_ask=max_time_ask, group_vars = copy.copy(groupby_var), append=append, config=config, verbose=verbose)
    for message in messages2:
        print(message)

    if return_combined is True:
        return combined_df
    else:
        return None


def gen_slurm_param_from_params(output_location, proj_dir, combined_df, messages=[], parameter_flag='params',min_num_to_run=5,
                          testmode=True, suffix='', proj_prefix='GISP2', default_calc_length=28,sample=None,
                          ntasks=36, max_time_ask=240, group_vars = None, append=False, config=None, verbose=False):

    parameter_dir = proj_dir / 'parameters'
    slurm_dir = proj_dir / 'slurm'


    for message in messages:
        print(message)

    messages = []
    parameter_ds = combined_df.to_dict(orient='records')


    if group_vars is None:
        group_vars = ['E', 'tau', 'lag', 'col_var_id']

    print('gen_slurm_param_from_params', group_vars)
    for group_vals, grp_df in combined_df.groupby(group_vars):


        # print('group_vars', group_vars, 'group_vals', group_vals)
        to_run_grp = grp_df[grp_df['to_run'] == True]
        # print('to_run_grp', len(to_run_grp))
        done_grp = grp_df[grp_df['to_run'] == False]
        # print('done_grp', len(done_grp))

        surr_nums = to_run_grp[to_run_grp['surr_num'] > 0]['surr_num'].unique()
        surr_vals = to_run_grp.surr_var.unique()
        if len(surr_nums) > 1:
            if len(surr_vals) > 1:
                surr_tag = '_surrmulti'
            else:
                surr_tag = f'_surr{surr_vals[0]}'
            # surr_tag = '_surr'
        else:
            surr_tag = ''

        # print('sample', to_run_grp['sample'].unique())
        if len(surr_nums) == 1:
            to_run_grp['sample']=250
            # print('non surrogate sample size set to 250')
        else:
            to_run_grp['sample']=100
            # print('surrogate sample size set to 100')


        tau_tag = ''
        tau_vals = grp_df.tau.unique()
        # if 'tau' in group_vars:
        if len(to_run_grp.tau.unique()) == 1:
                tau_tag = f'_tau{to_run_grp.tau.unique()[0]}'

        E_tag = ''
        E_vals = grp_df.E.unique()
        # if 'E' in group_vars:
        if len(to_run_grp.E.unique()) == 1:
                E_tag = f'_E{to_run_grp.E.unique()[0]}'


        lag_tag = '_lags'
        lag_vals = grp_df.lag.unique()
        # if 'lag' in group_vars:
        if len(to_run_grp.lag.unique()) == 1:
                lag_tag = f'_lag{to_run_grp.lag.unique()[0]}'

        col_tag = ''
        col_vars = grp_df.col_var_id.unique()
        # if 'col_var_id' in group_vars:
        if len(to_run_grp.col_var_id.unique()) == 1:
                col_tag = f'_{to_run_grp.col_var_id.unique()[0]}'


        target_tag = ''
        target_vars = grp_df.target_var_id.unique()
        # if 'target_var_id' in group_vars:
        if len(to_run_grp.target_var_id.unique()) == 1:
                target_tag = f'_{to_run_grp.target_var_id.unique()[0]}'

        if testmode is True:
            # if verbose is True:
            target_string = ','.join([str(tv) for tv in target_vars]).strip(',')
            col_string = ','.join([str(cv) for cv in col_vars]).strip(',')
            messages.append(
                      f'\n{target_string}, {col_string}; tau={tau_vals}, E={E_vals}, lag={lag_vals}')#, len={len(to_run_grp.groupby(["E", "tau", "Tp", "lag", "col_var_id", "surr_var", "surr_num"]))} unique combinations')


        param_tag = f'{lag_tag}{E_tag}{tau_tag}{col_tag}{target_tag}_{surr_tag}'
        param_tag = param_tag.strip('_')
        csv_name = f'{parameter_flag}{target_tag}_knn20{param_tag}{suffix}'  # _to16'
        source_file_path = proj_dir.parent/'templates' / 'hpc__run_ccm__lag.slurm'
        new_file_name = f'{proj_prefix}__a0_nodes_run_ccm_{param_tag}{suffix}.slurm'

        if append is True:
            new_file_name = f'{proj_prefix}__a0_nodes_run_ccm_{param_tag}_append.slurm'


        if len(to_run_grp) > min_num_to_run:

            if testmode is True:
                messages+=[
                    f'\t{csv_name}\n\t number of combs to run/unique: {len(to_run_grp)}/{len(grp_df)}, # completed: {len(done_grp)}', #_parameter_df.groupby(["E", "tau", "Tp", "lag", "col_var_id", "surr_var"]))} unique combinations, {len(_parameter_df)} parameter rows,  # done: {len(done)}')
                    f'\twould write to: {parameter_dir} / {csv_name}.csv',
                    f'\twould create slurm script: {new_file_name}']


            else:
                to_run_grp.to_csv(parameter_dir / f'{csv_name}.csv', index=False)
                make_slurm_script(to_run_grp, csv_name, new_file_name, slurm_dir, source_file_path,
                                  default_calc_length=default_calc_length, max_time_ask=max_time_ask,
                                  buffer_percent=1.5, ntasks=ntasks, append=append)

                print(f'slurm_submit.sh {proj_dir.name}/slurm/{new_file_name}')
        else:
            if len(to_run_grp) > 0:
                done_message = f'\tplease consolidate: fewer than {min_num_to_run} parameters to run for {csv_name}, skipping. # done: {len(done_grp)}, # total: {len(grp_df)}'
            else:
                done_message = f'\tNo parameters to run for {csv_name}, skipping. # done: {len(done_grp)}, # total: {len(parameter_ds)}'

            messages.append(done_message)

    return messages
