import copy
import os
import shutil
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from cedarkit.utils.workflow import get_assessed_param_picks
from cedarkit.utils.cli import setup_logging, log_line
from cedarkit.core.project_config import load_config
from cedarkit.utils.routing import set_calc_path
from cedarkit.utils.routing.paths import resolve_consolidated_dir, resolve_intermediate_dir


def _resolve_check_output_location(proj_dir, output_location, config, source, check):
    source = source or 'csv'
    if source not in ('csv', 'parquet'):
        raise ValueError(f"Unsupported source '{source}'. Expected 'csv' or 'parquet'.")

    check = check or 'intermediate'
    if check not in ('intermediate', 'consolidated'):
        raise ValueError(f"Unsupported check '{check}'. Expected 'intermediate' or 'consolidated'.")

    cfg = config
    if cfg is None:
        cfg = load_config(Path(proj_dir) / 'proj_config.yaml')

    calc_location = set_calc_path(None, Path(proj_dir), cfg)
    if check == 'intermediate':
        return resolve_intermediate_dir(calc_location, cfg, source)
    return resolve_consolidated_dir(calc_location, cfg, source)

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
                                          source='csv', check='intermediate', ntasks=42, max_time_ask=300, verbose= False):
    """Check which (E, tau) combinations still need runs, and emit slurm scripts for them.

    Two-stage pipeline: first resolves which output location to check for
    already-computed results (``_resolve_check_output_location``, via
    ``source``/``check``) and calls ``get_assessed_param_picks`` to filter
    ``comb_df`` down to what's still outstanding; then calls
    ``gen_slurm_param_from_params`` to write parameter CSVs and slurm scripts
    for those combinations, printing its status messages.

    Parameters
    ----------
    proj_dir : str or pathlib.Path
        Project root directory.
    output_location : str or pathlib.Path
        Directory where parameter/slurm files are written.
    comb_df : pandas.DataFrame
        Candidate parameter combinations to check and (if needed) run, as
        produced by ``make_comb_df``.
    min_num_to_run : int, default 8
        Minimum number of outstanding parameters in a group before a slurm
        script is actually generated for it; groups below this threshold are
        flagged as "please consolidate" instead.
    config : cedarkit.core.project_config.ProjectConfig, optional
        Project configuration. Loaded from ``proj_dir/proj_config.yaml`` if
        ``None``.
    parameter_dir : str or pathlib.Path, optional
        Directory for parameter files. Defaults to ``'parameters'`` under
        ``proj_dir`` if ``None``.
    surr : bool, default False
        Whether to include surrogate runs in the generated parameters.
    surr_num : int, default 201
        Number of surrogate draws to consider when ``surr`` is ``True``.
    groupby_var : list of str, optional
        Variables to group combinations by when checking completeness and
        writing scripts.
    testmode : bool, default True
        If ``True``, checks are performed but no slurm scripts are actually
        written.
    tp_vals : list of int, default [1]
        Prediction-horizon (``Tp``) values to sweep.
    knn_vals : list of int, default [20]
        Number-of-neighbors values to sweep.
    suffix : str, default ''
        Appended to generated parameter/script filenames.
    append : bool, default False
        Whether to append to existing parameter files rather than
        overwriting them.
    proj_prefix : str, default 'eevw'
        Prefix used in generated file names.
    default_calc_length : int, default 28
        Default requested slurm job length (see ``make_slurm_script``).
    surr_vars : list of str, optional
        Surrogate variable names to include, if different from the
        defaults inferred elsewhere.
    sample : int, default 150
        Row-count threshold passed through to ``get_assessed_param_picks``
        (as ``row_count_threshold``) for deciding a parameter set already
        has enough completed rows.
    return_combined : bool, default False
        If ``True``, return the ``combined_df`` produced by
        ``get_assessed_param_picks`` instead of ``None``.
    source : {'csv', 'parquet'}, default 'csv'
        Output format to check for existing results.
    check : {'intermediate', 'consolidated'}, default 'intermediate'
        Whether to check per-run intermediate output or the consolidated
        output directory.
    ntasks : int, default 42
        Slurm ``--ntasks`` value written into generated scripts.
    max_time_ask : int, default 300
        Maximum slurm walltime (minutes) requested in generated scripts.
    verbose : bool, default False
        Whether to print verbose status output.

    Returns
    -------
    pandas.DataFrame or None
        ``combined_df`` if ``return_combined`` is ``True``, otherwise
        ``None`` — regardless of ``testmode``.

    See Also
    --------
    make_comb_df : Builds the ``comb_df`` this function consumes.
    """
    print('gen_parameters_slurm2', groupby_var)
    check_output_location = _resolve_check_output_location(proj_dir, output_location, config, source, check)
    print(f'checking existing outputs in: {check_output_location} (source={source}, check={check})')
    # calls get_assessed_param_picks internally
    combined_df, messages = get_assessed_param_picks(proj_dir, check_output_location, comb_df,config=config,parameter_dir=parameter_dir, surr=surr, surr_num=surr_num, groupby_vars = copy.copy(groupby_var), testmode=testmode,
                            tp_vals = tp_vals, knn_vals = knn_vals,append=append,surr_vars=surr_vars, verbose= verbose, source=source, row_count_threshold=sample)
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
    print('combined_df', combined_df.head())
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
