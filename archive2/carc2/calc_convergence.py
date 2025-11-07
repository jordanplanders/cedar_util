import datetime, time
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd
pd.option_context('mode.use_inf_as_na', True)

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import t
from scipy import stats

from utils.arg_parser import get_parser, parse_flags
from utils.location_helpers import *
from utils.config_parser import load_config
from utils.data_access import rel_reformat, relationship_filter, get_real_output, get_weighted_flag
# from utils.data_processing import relationship_filter, rel_reformat
from archive2.utils.run_tools import decide_file_handling

import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def f(x, bmax, x0, kd, rho_0):
    # Ensure that L is always greater than or equal to L_0
    L_adjusted = np.maximum(x - x0, 0)
    return rho_0 + (bmax - rho_0) * (L_adjusted) / (kd + L_adjusted)

def filter_ptile(group, l=.25, u=.75):
    Q1 = group['rho'].quantile(l)
    Q3 = group['rho'].quantile(u)
    IQR = Q3 - Q1
    return group[(group['rho'] >= Q1) & (group['rho'] <= Q3)]


def fit_binding(f, x_data, y_data, initial_guess, bounds):
    popt, pcov = curve_fit(f, x_data, y_data, p0=initial_guess, bounds=bounds)
    bmax_opt, x0_opt, kd_opt, rho_0_opt = popt

    param_errors = np.sqrt(np.diag(pcov))
    # print(f"Standard errors:\nL = {param_errors[0]}\nk = {param_errors[1]}\nx0 = {param_errors[2]}")

    # Calculate confidence intervals
    alpha = 0.05  # 05  # 95% confidence interval
    n = len(y_data)  # number of data points
    p = len(popt)  # number of parameters
    dof = max(0, n - p)  # degrees of freedom

    # t-value for the confidence interval
    t_val = t.ppf(1 - alpha / 2, dof)

    # Confidence intervals
    ci = param_errors * t_val

    fit_results = {
        "max_lib_size": max(x_data),
        "min_lib_size": min(x_data),
        "L": bmax_opt,
        "Kd": kd_opt,
        "x0": x0_opt,  # min(lib_sizes),
        "off": rho_0_opt,  # rho_0,
        "L_error": param_errors[0],
        "Kd_error": param_errors[1],
        "x0_error": param_errors[2],
        "L_CI": (bmax_opt - ci[0], bmax_opt + ci[0]),
        "Kd_CI": (kd_opt - ci[1], kd_opt + ci[1]),
        "x0_CI": (x0_opt - ci[2], x0_opt + ci[2]),

    }
    return popt, pcov, fit_results


def plot_convergence(xedges, x_fitted, yedges, y_fitted, y_fitted_upper, middle_df, convergence_interval,
                     convergence_ts, popt_upper, popt_middle, behave_d, title, fig_name, color='r', cmap='plasma', figsize=(12, 6), h=None):


    bmax_opt_upper, x0_opt_upper, kd_opt_upper, rho_0_opt_upper = popt_upper
    bmax_opt, x0_opt, kd_opt, rho_0_opt = popt_middle

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    pcm1 = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm())  # Log scale
    fig.colorbar(pcm1, ax=ax, label='# points')
    ax.set_xlabel('LibSize')
    ax.set_ylabel(r'$\rho$')

    # Plot the annotations
    ax.axvline(convergence_ts, color=color, linestyle='--',
               label=f'convergence ts={convergence_ts:.2f}')
    ax.axvline(middle_df.LibSize.max(), color='teal', linestyle='--',
               label=f'upper limit of centered binding curve {middle_df.LibSize.max()}')

    ax.plot(x_fitted, y_fitted, color=color,
            label=f"central binding curve; ct={kd_opt * 7 + 20:.2f}, rho_f={bmax_opt:.2f}")
    ax.plot(x_fitted, y_fitted_upper, color='k', linestyle='--', alpha=.5,
            label=f">= 90%ile binding curve; ct={kd_opt_upper * 7 + 20:.2f}, rho_f={bmax_opt_upper:.2f}")
    ax.set_xlim([0, 350])

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_title('\n'.join([element for element in [title, behave_d['top_behavior'],
                                                    behave_d['core_behavior'],
                                                    behave_d['end_behavior']] if
                            len(element) > 0]).strip('\n'))
    if 'erb' in fig_name:
        ax.set_ylim([-.15, .7])
    else:
        ax.set_ylim([-.15, .4])

    if convergence_interval is not None:
        if len(behave_d['end_behavior']) > 0:
            alpha = .5
        else:
            alpha = .2
        ax.axvspan(convergence_interval[0], convergence_interval[1], color='k', alpha=alpha,
                   label='convergence interval')

    return fig, ax


# def streamline_cause(label, split_word='causes'):
#     label_parts = label.split(' {} '.format(split_word))
#     clean_parts = []
#     for part in label_parts:
#         # print(part.split('(t-'))
#         try:
#             part = ' '.join(list(set([col.split(')')[-1].strip(' ') for col in part.split('(t-')]))).strip(' ')
#         except:
#             part = part.strip(' ')
#         clean_parts.append(part)
#     label = f' {split_word} '.join(clean_parts)
#     return label

# # Function to fetch and prepare real data
# def get_real_data(real_dfs_references, grp_path, meta_variables, max_libsize, knn, sample_size=500):
#     if isinstance(real_dfs_references,list) ==True:
#         if len(real_dfs_references) == 0:
#             print('No real data found', file=sys.stderr, flush=True)
#             return None
#         elif len(real_dfs_references) > 1:
#             # files = [real_dfs_references[0]]
#             real_df_full = pd.concat([pd.read_csv(grp_path / file) for file in real_dfs_references])
#             print('Multiple files found, concatenating:', len(real_dfs_references), file=sys.stderr, flush=True)
#         else:
#             files = real_dfs_references
#             real_df_full = pd.read_csv(grp_path / files[0])
#         # print(real_df_full.head(), file=sys.stderr, flush=True)
#     else:
#         real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)
#
#     real_df_full = real_df_full[real_df_full['LibSize'] >= knn].copy()
#     if real_df_full.empty:
#         return None
#
#     real_df = real_df_full[real_df_full['LibSize'] <= max_libsize].copy()
#     rel_dfs = [
#         rel_df.groupby('LibSize').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
#         for _, rel_df in real_df.groupby('relation')
#     ]
#     real_df = pd.concat(rel_dfs).reset_index(drop=True)
#     real_df['surr_var'] = 'neither'
#
#     try:
#         real_df['relation'] = real_df['relation'].apply(lambda x: streamline_cause(x))
#     except Exception as e:
#         print(f'Error applying streamline_cause: {e}', file=sys.stderr)
#
#     return real_df

# def remove_numbers(input_string):
#     # Use regex to remove all digits from the string
#     return re.sub(r'\d+', '', input_string)


# # Function to fetch and prepare surrogate data
# def get_surrogate_data(surr_dfs_references, grp_path, meta_variables, max_libsize, knn, _rel,
#                        sample_size=400):
#     surr_dfs = []
#     ctr = 0
#     for df_csv_name in surr_dfs_references:
#         surr_df_full = pd.read_csv(grp_path / df_csv_name)
#     # for pset_id, pset_df in surr_dfs_references.groupby('pset_id'):
#     #     surr_df = collect_raw_data(pset_df, meta_vars=meta_variables)
#         surr_df = surr_df_full[surr_df_full['relation'] == _rel].copy()
#         if surr_df.empty:
#             continue
#
#         if 'surr_var' not in surr_df.columns:
#             surr_var = remove_numbers(df_csv_name.split('__')[1].split('.csv')[0])
#             if surr_var == 'tsi':
#                 surr_var = 'TSI'
#             surr_df['surr_var'] = surr_var
#
#         surr_df = surr_df[(surr_df['surr_var'] != 'neither') & (surr_df['LibSize'] >= knn) & (surr_df['LibSize'] <= max_libsize)].copy()
#         rel_dfs = [
#             rel_df.groupby('LibSize').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
#             for _, rel_df in surr_df.groupby('surr_var')
#         ]
#         surr_df = pd.concat(rel_dfs).reset_index(drop=True)
#         rel_dfs = []
#         for surr_var, surr_df_i in surr_df.groupby('surr_var'):
#             surr_df_i['relation_s'] = surr_df_i['relation'].str.replace(surr_var, f'{surr_var} (surr) ')
#             surr_df_i['relation_s'] =surr_df_i['relation_s'].str.strip()
#             surr_df_i['relation_s'] = surr_df_i['relation_s'].str.replace('  ', ' ')
#             rel_dfs.append(surr_df_i)
#         surr_df = pd.concat(rel_dfs).reset_index(drop=True)
#         # surr_df['relation_s'] = surr_df.apply(lambda row: row['relation'].replace(row['surr_var'], f'{row["surr_var"]} (surr)'), axis=1)
#         if len(surr_df)>0:
#             surr_dfs.append(surr_df)
#             ctr += 1
#         # if ctr > 3:
#         #     continue
#
#     if len(surr_dfs) == 0:
#         print('No surrogate data found', file=sys.stderr, flush=True)
#         return None
#     else:
#         surr_df_full = pd.concat(surr_dfs).reset_index(drop=True)
#         return surr_df_full


def process_group(arg_tuple):

    filter_center = lambda x: filter_ptile(x, .1, .9)
    filter_high = lambda x: filter_ptile(x, .9, .975)

    (grp_d, ind, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
     function_flag, plot_flag, args, percent_threshold, grp_path) = arg_tuple

    # print('override_flag', override_flag)

    start_time = datetime.datetime.now()
    datetime_flag = args.datetime_flag
    print(f'grp_id:{str(grp_d["group_id"])} received', start_time, file=sys.stdout, flush=True)
    # calc_metric_csvs = calc_convergence_dir /config.calc_convergence_dir.dirs.csvs  # 'csvs2'
    # metric_df_path = calc_metric_csvs / f'{str(grp_d["group_id"])}.csv'
    fig_dir = convergence_dir_csv_parts.parent /config.calc_convergence_dir.dirs.figures  # calc_metric_dir / 'figures2'
    spec_fig_dir = fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' #/f'lag{grp_d["lag"]}'
    spec_fig_dir.mkdir(exist_ok=True, parents=True)

    weighted_flag = get_weighted_flag(grp_d)
    pal = config.pal.to_dict()
    surr_testing_window=75
    _max_libsize = 325

    metric = 'L_convergence_detail'
    fig_name = f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}__libsize_rho_{metric}__{str(grp_d["group_id"])}.png'
    fig_exists = (spec_fig_dir / fig_name).exists()
    run_continue, overwrite = decide_file_handling(args, fig_exists)
    if run_continue is False:
        print(f'!\tfigure: grp_id:{str(grp_d["group_id"])} already exists, skipping', file=sys.stdout, flush=True)
        return

    snippet_csv = convergence_dir_csv_parts / 'convergence_grp{}_ind{}.csv'.format(int(grp_d['group_id']), ind)
    existing_csv = [convergence_dir_csv_parts / file for file in os.listdir(convergence_dir_csv_parts) if
                    file.startswith('convergence_grp{}_ind'.format(int(grp_d['group_id'])))]
    run_continue, overwrite_snippets = decide_file_handling(args, len(existing_csv) > 0)

    # if override_flag in [False, 'False']:
    #     if write_flag in [False, 'False', None, 'append']:
    #         if (spec_fig_dir / fig_name).exists():
    #             print(f'figure: grp_id:{str(grp_d["group_id"])} already exists, skipping', file=sys.stdout, flush=True)
    #             return

    # # Check if the metric dataframe already exists and if it should be overridden
    # metric_df_exists = metric_df_path.exists()
    # dt_m = datetime.datetime.fromtimestamp(os.path.getmtime(metric_df_path)) if metric_df_exists else None
    # run_continue, overwrite = decide_file_handling(args, metric_df_exists, modify_datetime = dt_m)
    # # choice = check_for_exit(True, datetime_flag, metric_df_path, grp_d)

    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']

    ik = 0
    while ik < 10:
        real_df_full = get_real_output(real_dfs_references, grp_path, meta_variables, _max_libsize, grp_d['knn'])
        if real_df_full is None:
            print('No real data found', grp_d, file=sys.stderr, flush=True)
            if ik == 9:
                return None
            else:
                ik += 1
        else:
            ik = 10
    if len(real_df_full) == 0:
        print('no real data found', grp_d, file=sys.stderr, flush=True)
        return
    real_df_full = real_df_full[real_df_full['LibSize'] >= grp_d['knn']].copy()
    _real_df = real_df_full[real_df_full['LibSize'] <= _max_libsize].copy()


    target_relation = None
    if config.has_nested_attribute('target_relation'):
        target_relation = config.target_relation
        real_df = relationship_filter(_real_df, target_relation) #surr_df_full[surr_df_full['relation'].isin([_rel, _rel.replace('influences', 'causes')])].copy()
        real_df = rel_reformat(real_df, 'relation')
        #
        # real_df = _real_df[_real_df['relation'].isin([target_relation, target_relation.replace('influences', 'causes')])].copy()
        # real_df['relation'] = real_df['relation'].apply(lambda x: x.replace('causes','influences'))

    real_df = real_df[~real_df['relation'].str.isnumeric()].copy()

    if len(real_df) < 1:
        end_time = datetime.datetime.now()
        print(f'data for available: {len(real_df)}, grp_id: {grp_d["group_id"]}', grp_d, 'time elapsed: ',
              end_time - start_time, file=sys.stderr, flush=True)
        return

    fit_list = []
    for rel, rel_df in real_df.groupby('relation'):

        if len(rel_df) < 10:
            print('not enough data for real df', rel, file=sys.stderr, flush=True)
            continue
        else:
            # try:
            rel_df[['rho', 'LibSize']] = rel_df[['rho', 'LibSize']].astype(float)
            def_len = 700
            bounds = ([-np.inf, -np.inf, 1, -np.inf], [np.inf, np.inf, np.inf, np.inf])
            if 'erb' in [grp_d['col_var_id'], grp_d['target_var_id']]:
                initial_guess = [.4, 19, 25, .4]
            else:
                initial_guess = [.2, 19, 25, .2]

            libsize_max=325
            color = pal[rel]
            stop=False
            for ip in range(5):
                if stop is True:
                    print('stopping', file=sys.stderr, flush=True)
                    continue
                else:
                    if ip>0:
                        fit_list = []

                    rel_df = rel_df.groupby('LibSize').apply(lambda x: x.sample(n=800, replace=True)).reset_index(drop=True)

                    flag_d = {'end_behavior': [], 'core_behavior': [], 'top_behavior': [], 'interval': []}

                    ## fit upper side of envelope
                    filtered_df = rel_df[(rel_df['LibSize'] <= 250)].copy()
                    sample_df = filtered_df.groupby('LibSize').apply(filter_high).reset_index(drop=True)
                    x_data = sample_df['LibSize']
                    y_data = sample_df['rho']

                    # Fit the function to the data
                    try:
                        popt_upper, pcov_upper, fit_results_upper = fit_binding(f, x_data, y_data, initial_guess, bounds)
                    except:
                        print(f'error with curve fit, E={grp_d["E"]}, tau={grp_d["tau"]}', file=sys.stderr, flush=True)
                        continue
                    x_fitted = np.arange(0, 350)  # EW_test.var1_xmap_var2['LibSize'].unique()
                    y_fitted_upper = f(x_fitted, *popt_upper)

                    # Extract the optimized parameters
                    bmax_opt_upper, x0_opt_upper, kd_opt_upper, rho_0_opt_upper = popt_upper
                    print(
                        f"Optimized parameters from above: bmax: {bmax_opt_upper}, x0: {x0_opt_upper}, kd: {kd_opt_upper}, rho_0: {rho_0_opt_upper}")
                    if rho_0_opt_upper * .975 >= bmax_opt_upper * 1.025:
                        flag_d['top_behavior'].append('converge from above')
                    elif f(x_fitted[0], *popt_upper) * .975 >= bmax_opt_upper * 1.025:
                        flag_d['top_behavior'].append('converge from above')

                    fit_results_upper.update(grp_d)
                    fit_results_upper['curve_fit'] = 'upper_envelope'
                    fit_results_upper['relation'] = rel
                    for flag in flag_d:
                        fit_results_upper[flag]='\n'.join(flag_d[flag]).strip('\n') if len(flag_d[flag])>0 else None
                    fit_list.append(fit_results_upper)

                    ## Fit the middle of the envelope
                    # Group by 'LibSize' and apply the filter
                    filtered_df = rel_df[(rel_df['LibSize'] <= libsize_max)].copy()
                    sample_df = filtered_df.groupby('LibSize').apply(filter_center).reset_index(drop=True)
                    x_data = sample_df['LibSize']
                    y_data = sample_df['rho']

                    # Fit the function to the data
                    try:
                        popt_middle, pcov_middle, fit_results_middle = fit_binding(f, x_data, y_data, initial_guess, bounds)
                    except:
                        print(f'error with middle curve fit, E={grp_d["E"]}, tau={grp_d["tau"]}', file=sys.stderr, flush=True)
                        continue

                    fit_results_middle.update(grp_d)
                    fit_results_middle['curve_fit'] = 'middle_envelope'
                    fit_results_middle['relation'] = rel

                    # Extract the optimized parameters
                    bmax_opt, x0_opt, kd_opt, rho_0_opt = popt_middle

                    ## Evaluate
                    libsizes, rel_conv, rel_curve, rel_curve_interval = [], [], [], []
                    init_rho = f(20, *popt_middle)
                    for libsize in sample_df['LibSize'].unique():
                        rhos = sample_df[sample_df['LibSize'] == libsize].rho.values
                        libsizes.append(libsize)
                        rel_conv.append(len(rhos[rhos > bmax_opt]) / len(rhos))
                        rel_curve.append(len(rhos[rhos > f(libsize, *popt_middle)]) / len(rhos))
                        est_rho = f(libsize, *popt_middle)
                        delta_rho = np.abs(est_rho - init_rho) * percent_threshold
                        filtered_rhos = rhos[rhos > est_rho-delta_rho]
                        filtered_rhos = filtered_rhos[filtered_rhos < est_rho+delta_rho]
                        rel_curve_interval.append(len(filtered_rhos) / len(rhos))

                    metrics_df = pd.DataFrame({'LibSize': libsizes, 'rel_conv': rel_conv, 'rel_curve': rel_curve, 'rel_curve_interval':rel_curve_interval})
                    middle_df = metrics_df[((metrics_df['rel_curve'] >= .375) & (metrics_df['rel_curve'] <= .625)) | (metrics_df['rel_curve_interval']>=.50)].copy()
                    middle_fit_outer_bound = middle_df.LibSize.max()

                    convergence_ts = 7 * kd_opt + 20
                    y_fitted = f(x_fitted, *popt_middle)

                    ## Plotting
                    bins_x = len(filtered_df['LibSize'].unique())  # Number of bins for libsize (adjust as needed)
                    bins_y = 200  # Number of bins for rho (adjust as needed)
                    rho = filtered_df['rho'].values
                    libsize = filtered_df['LibSize'].values
                    title = f'E={grp_d["E"]}, tau={grp_d["tau"]}'

                    # Evaluate the convergence
                    counter = 0
                    while counter <2:
                        caveats = []
                        if counter ==0:
                            large_lib_rho = filtered_df[filtered_df['LibSize'] >= libsize_max - 20].rho
                            greater_than_zero = large_lib_rho.values[large_lib_rho.values > 0]
                        if counter ==1:
                            large_lib_rho = filtered_df[(filtered_df['LibSize'] >= middle_fit_outer_bound-20) &
                                                        (filtered_df['LibSize'] <= middle_fit_outer_bound)].rho
                            greater_than_zero = large_lib_rho.values[large_lib_rho.values > 0]

                        # no improvement
                        small_lib_rhos = filtered_df[filtered_df['LibSize'] < 40].rho.values[
                                filtered_df[filtered_df['LibSize'] < 40].rho.values > 0]
                        res = stats.ttest_ind(
                            greater_than_zero,
                            small_lib_rhos,
                            equal_var=False, alternative='greater')
                        if res.pvalue > .05:
                            caveats.append(
                                f'end behavior same mean, {np.mean(greater_than_zero):.3f}, {np.mean(small_lib_rhos):.3f}')
                            print(
                                f'p value, same mean, {np.mean(greater_than_zero):.3f}, {np.mean(small_lib_rhos):.3f}', res.pvalue)

                        # final rho mean not statistically different from zero
                        res = stats.ttest_ind_from_stats(large_lib_rho.mean(),
                                                         large_lib_rho.std(),
                                                         large_lib_rho.count(),
                                                         0,
                                                         large_lib_rho.std(),
                                                         large_lib_rho.count(), equal_var=True, alternative='greater')

                        if res.pvalue > .05:
                            caveats.append(f'end behavior, zero mean, {large_lib_rho.mean():.3f}')
                            print(f'p value, zero mean, {large_lib_rho.mean():.3f}', res.pvalue)
                        else:
                            res = stats.ttest_1samp(large_lib_rho.values, popmean=0, alternative='greater')
                            if res.pvalue > .05:
                                caveats.append('end behavior, zero popmean')
                                print('p value, zero mean', res.pvalue)
                            else:
                                # final rho mean not statistically different from zero
                                perc_geq_zero = len(greater_than_zero) / len(large_lib_rho)
                                if perc_geq_zero < .90:
                                    caveats.append(f'end behavior, perc greater than zero, {perc_geq_zero:.3f}')
                                    print('perc greater than zero', perc_geq_zero)

                        if len(caveats) > 0:
                            print('caveats', caveats)
                            if counter == 0:
                                flag_d['end_behavior'] = caveats
                            if counter == 1:
                                flag_d['core_behavior'] = caveats
                                counter=2
                            counter += 1
                        else:
                            counter = 2

                    end_behavior = ',\n'.join(flag_d['end_behavior']).strip(',\n') if len(flag_d['end_behavior']) > 0 else ''
                    core_behavior = ',\n'.join(flag_d['core_behavior']).strip(',\n') if len(flag_d['core_behavior']) > 0 else ''
                    top_behavior = ',\n'.join(flag_d['top_behavior']).strip(',\n') if len(flag_d['top_behavior']) > 0 else ''
                    behave_d = {'end_behavior': end_behavior, 'core_behavior': core_behavior, 'top_behavior': top_behavior}

                    cut_value = 25

                    tmp = middle_df.copy()#metrics_df[(metrics_df['rel_curve'] > .375) & (metrics_df['rel_curve'] < .625) & (
                                # metrics_df['LibSize'] <= middle_df.LibSize.max())].copy()
                    tmp = tmp.reset_index(drop=True)
                    tmp['delta'] = tmp['LibSize'].diff()

                    convergence_interval=None
                    libsizes = tmp['LibSize'].sort_values(ascending=False).values
                    for libsize in libsizes:
                        if convergence_interval is None:
                            less_than_df = tmp[tmp['LibSize']<=libsize].copy()
                            lower_edge_LibSize = libsize - surr_testing_window#less_than_df[
                                # less_than_df['LibSize'] <= (libsize - surr_testing_window)].LibSize.max()
                            less_than_df = less_than_df[(less_than_df['LibSize'] >= lower_edge_LibSize)].copy()
                            if less_than_df["delta"].mean() <= cut_value:
                                convergence_interval = (
                                    less_than_df.LibSize.max() - surr_testing_window, less_than_df.LibSize.max())
                                title += f'\nConvergence time: {convergence_ts:.2f}, average delta {less_than_df["delta"].mean():.3f}'
                            else:
                                print('E', grp_d['E'], 'tau', grp_d['tau'], 'less than', libsize, lower_edge_LibSize, less_than_df["delta"].mean())

                    if convergence_interval is None:
                        title += f'\nConvergence time: {convergence_ts:.3f} (interval too short), average delta {less_than_df["delta"].mean():.3f}'
                        flag_d['interval'].append('no convergence interval')


                    fit_results_middle['convergence_interval'] = convergence_interval

                    for flag in flag_d:
                        fit_results_middle[flag]='\n'.join(flag_d[flag]).strip('\n') if len(flag_d[flag])>0 else None
                    fit_results_middle.update(grp_d)
                    fit_list.append(fit_results_middle)

                    flag_count = sum([len(flag_d[flag]) for flag in ['end_behavior', 'core_behavior', 'top_behavior']])
                    if (convergence_interval is None) & (flag_count == 0):
                        stop = False
                    else:
                        stop = True
                        if len(fit_list)==2:
                            title = '[assessed] ' + title

                rho = filtered_df['rho'].values
                libsize = filtered_df['LibSize'].values
                h, xedges, yedges = np.histogram2d(libsize, rho, bins=[bins_x, bins_y])

                fig, ax = plot_convergence(xedges, x_fitted, yedges, y_fitted, y_fitted_upper, middle_df,
                                 convergence_interval, convergence_ts, popt_upper, popt_middle,
                                 behave_d, title, fig_name, color=color, cmap='plasma', h=h)


    if convergence_interval is None:
        flag_count += 1

    if flag_count > 0:
        spec_fig_dir = spec_fig_dir / 'fail'
    else:
        spec_fig_dir = spec_fig_dir / 'pass'

    spec_fig_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(spec_fig_dir / fig_name, dpi=300, bbox_inches='tight')
    print('fig saved', spec_fig_dir / fig_name, file=sys.stdout, flush=True)
    plt.close()

    # snippet_csv = convergence_dir_csv_parts / 'convergence_grp{}_ind{}.csv'.format(int(grp_d['group_id']), ind)
    # existing_csv = [convergence_dir_csv_parts/file for file in os.listdir(convergence_dir_csv_parts) if
    #                 file.startswith('convergence_grp{}_ind'.format(int(grp_d['group_id'])))]
    # run_continue, overwrite = decide_file_handling(args, len(existing_csv)>0)

    if len(fit_list) > 0:
        print('convergence_grps', snippet_csv, file=sys.stdout, flush=True)
        new_data_df = pd.DataFrame(fit_list)
        new_data_df = write_to_file(new_data_df, snippet_csv, existing_csv, overwrite=overwrite_snippets)
        # #
        # # if write_flag in ['replace']:#[True, 'True']:
        # #     new_data_df.to_csv(snippet_csv)
        # else:
        #     snippets = []
        #     if len(existing_csv) > 0:
        #         for csv in existing_csv:
        #             snippets.append(pd.read_csv(csv, index_col=0))
        #     snippets.append(new_data_df)
        #     pd.concat(snippets).to_csv(snippet_csv, index=False)
        return new_data_df
    else:
        return

def write_to_file(ccm_out_df, df_path, existing_csvs=None, overwrite=False):
    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len',
                   'train_ind_i']
    ccm_out_df = ccm_out_df[[col for col in ccm_out_df.columns if col not in remove_cols]].copy()

    try:
        ccm_out_df_0 = pd.concat([pd.read_csv(csv_path, index_col=0) for csv_path in existing_csvs])
        if overwrite == False:
            ccm_out_df_0 = ccm_out_df_0[[col for col in ccm_out_df_0.columns if col not in remove_cols]].copy()
            ccm_out_df = pd.concat([ccm_out_df_0, ccm_out_df])
            ccm_out_df.reset_index(drop=True, inplace=True)
    except Exception as e:
        pass

    ccm_out_df.to_csv(df_path)
    print('!\twrote to file: ', df_path)
    return ccm_out_df

if __name__ == '__main__':

    delta_label = 'r50p-s50p'
    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # override = args.override
    # write = args.write
    # if override is False:
    #     if write in ['append', None]:
    #         write = 'append'

    second_suffix = f'_{int(time.time() * 1000)}' if args.test is True else ''

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    # if Path('/Users/jlanders').exists() == True:
    #     calc_location = proj_dir / config.local.calc_carc
    # else:
    #     calc_location = proj_dir / config.carc.calc_carc

    calc_location = set_calc_path(args, proj_dir, config, second_suffix)
    output_dir = set_output_path(args, calc_location, config)


    percent_threshold, function_flag, res_flag, second_suffix = parse_flags(args,
                                                                                 default_percent_threshold=.05,
                                                                                 default_function_flag='binding',
                                                                                 default_res_flag='',
                                                                                 default_second_suffix=second_suffix)

    calc_convergence_dir_name = construct_convergence_name(args, carc_config_d, percent_threshold, second_suffix)
    calc_convergence_dir_parent = calc_location / calc_convergence_dir_name
    calc_convergence_dir_parent.mkdir(exist_ok=True, parents=True)
    #
    # calc_convergence_dir_csvs = calc_convergence_dir /config.calc_convergence_dir.dirs.csvs
    # calc_convergence_dir_csvs.mkdir(exist_ok=True, parents=True)
    #
    # convergence_dir_csv_parts = calc_convergence_dir / f'{config.calc_convergence_dir.dirs.summary_frags}'
    # convergence_dir_csv_parts.mkdir(exist_ok=True, parents=True)
    #
    # fig_dir = calc_convergence_dir /f'{config.calc_convergence_dir.dirs.figures}'
    # fig_dir.mkdir(exist_ok=True, parents=True)

    query_keys = config.calc_criteria_rates2.query_keys
    if 'knn' not in query_keys:
        query_keys.append('knn')

    if args.group_file is not None:
        grp_csv = f'{args.group_file}.csv'
    else:
        grp_csv = f'{carc_config_d.csvs.calc_grp_run_csv}.csv'

    calc_grps_path = calc_carc_mirrored / grp_csv
    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()

    plot_flag = True
    # datetime_flag = None  #datetime.datetime(2024, 8, 31,6, 20 )

    # Determine the number of CPUs to use from the SLURM environment variable
    if Path('/Users/jlanders').exists():
        print('running locally', file=sys.stdout, flush=True)

        convergence_grps = []
        delta_grps = []
        calc_grps_df = calc_grps_df[
                                    # (calc_grps_df['col_var_id'] == 'erb') &
                                    (calc_grps_df['target_var_id'] == 'wu') &
                                    (calc_grps_df['train_ind_i'] == 0) &
                                    (calc_grps_df['E'] == 4) &
                                    (calc_grps_df['Tp'] == 20)].copy()
        ind = 0
        for authors, auth_grp in calc_grps_df.groupby(['col_var_id', 'target_var_id']):
            print(f'authors: {authors}', file=sys.stdout, flush=True)
            auth_pair = f'{authors[0]}_{authors[1]}'
            calc_grps = auth_grp.to_dict(orient='records')

            # calc_log2_df = calc_log2_df[~calc_log2_df['df_path'].str.contains('_pctiles')]
            arg_tuples = []

            for grp_d in calc_grps:
                grp_path = set_grp_path(output_dir, grp_d) #output_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'E{grp_d["E"]}_tau{grp_d["tau"]}'
                files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
                real_dfs_references = [file for file in files if 'neither' in file]
                if len(real_dfs_references)>0:
                    calc_convergence_dir, calc_convergence_dir_csvs, convergence_dir_csv_parts, fig_dir = set_convergence_paths(calc_convergence_dir_parent, config, grp_d)
                    arg_tuples.append(
                        (grp_d, ind, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
                         function_flag,
                         plot_flag, args, percent_threshold, grp_path))
                    ind+=1

            num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([4, len(arg_tuples)])))
            with Pool(num_cpus) as pool:
                results= pool.map(process_group, arg_tuples)

            results = [result for result in results if result is not None]

            if len(results) > 0:
                convergence_grps.append(pd.concat(results))

    else:
        if args.inds is not None:
            index = int(args.inds[-1])
        else:
            print('calc_criteria_rates2; index is required', file=sys.stderr, flush=True)
            sys.exit(0)

        grp_ds = calc_grps_df.to_dict(orient='records')
        if index >= len(grp_ds):
            print('index out of range', file=sys.stdout, flush=True)
            sys.exit(1)
        else:
            grp_d = grp_ds[index]

        # calc_log2_df = pd.read_csv(calc_carc_mirrored / 'calc_log4.csv', index_col=0, low_memory=False)
        # calc_log2_df = calc_log2_df[~calc_log2_df['df_path'].str.contains('_pctiles')]
        arg_tuples = []

        query_d = {}
        for key in query_keys:
            if isinstance(grp_d[key], str):
                query_d[key] = f'"{grp_d[key]}"'
            else:
                query_d[key] = grp_d[key]

        # query_string = [f'{key}=={value}' for key, value in query_d.items()]
        # query_string = ' & '.join(query_string)
        # grp_df = calc_log2_df.query(query_string)

        grp_path = set_grp_path(output_dir, grp_d)#output_dir/ f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'E{grp_d["E"]}_tau{grp_d["tau"]}'
        files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
        real_dfs_references = [file for file in files if 'neither' in file]
        surr_dfs_references = [file for file in files if 'neither' not in file]

        # real_dfs_references = grp_df[(grp_df['surr_var'] == 'neither') & (grp_df['weighted'].isin([False]))].copy()
        if len(real_dfs_references) > 0:
            calc_convergence_dir, calc_convergence_dir_csvs, convergence_dir_csv_parts, fig_dir = set_convergence_paths(calc_convergence_dir_parent, config, grp_d)

            # real_dfs_references=None
            arg_tuples.append(
                (grp_d, index, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
                 function_flag,
                 plot_flag, args, percent_threshold, grp_path))

        process_group(arg_tuples[0])
        result = process_group(arg_tuples[0])

        if len(result) > 0:
            print('convergence calculated successfully!', file=sys.stdout, flush=True)
        else:
            print('convergence not calculated', file=sys.stdout, flush=True)

    #SBATCH -p debug
