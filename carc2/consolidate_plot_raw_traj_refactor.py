# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from scipy import stats
# import datetime

import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import re
import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.data_access import collect_raw_data, get_weighted_flag, set_df_weighted, write_query_string
from utils.data_access import pull_percentile_data, get_group_sizes, get_sample_rep_n, check_empty_concat
from scipy import stats
from utils.data_processing import is_float
import ast

#
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
#
# ############ Data Grab
# # Function to fetch and prepare real data
# def get_real_data(real_dfs_references, meta_variables, max_libsize, knn, sample_size=500):
#     real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)
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
#
# # Function to fetch and prepare surrogate data
# def get_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, _rel,
#                        sample_size=400):
#     surr_dfs = []
#     ctr = 0
#     for pset_id, pset_df in surr_dfs_references.groupby('pset_id'):
#         surr_df = collect_raw_data(pset_df, meta_vars=meta_variables)
#         surr_df = surr_df[surr_df['relation'] == _rel].copy()
#         if surr_df.empty:
#             continue
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
#     if len(surr_dfs) == 0:
#         print('No surrogate data found', file=sys.stderr, flush=True)
#         return None
#     else:
#         surr_df_full = pd.concat(surr_dfs).reset_index(drop=True)
#         return surr_df_full
#
# # IQR filtering function
# def filter_ptile(group, l=0.25, u=0.75):
#     Q1 = group['rho'].quantile(l)
#     Q3 = group['rho'].quantile(u)
#     return group[(group['rho'] >= Q1) & (group['rho'] <= Q3)]
#
# # Function to construct delta rho dataframe
# def construct_delta_rho_df(real_df_full, surr_df_full, target_relation, conv_match_d, sample_size=500, pctile_range=[.25, .75]):
#     if 'convergence_interval' not in conv_match_d:
#         print('No convergence interval found', file=sys.stderr, flush=True)
#         return None
#
#     (interval_min, interval_max) = conv_match_d['convergence_interval']
#     print('interval_min', interval_min, 'interval_max', interval_max)
#     real_df = real_df_full[(real_df_full['LibSize'] >= interval_min) & (real_df_full['LibSize'] <= interval_max)].copy()
#     surr_df = surr_df_full[(surr_df_full['LibSize'] >= interval_min) & (surr_df_full['LibSize'] <= interval_max)].copy()
#
#     real_df_min = real_df_full[(real_df_full['LibSize'] < 40) & (real_df_full['relation'] == target_relation)].copy()
#     real_df_min = real_df_min.groupby('LibSize').apply(filter_ptile).reset_index(drop=True)
#     real_df_min = real_df_min.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
#
#     delta_rho_dfs = []
#     for libsize, real_df_libsize in real_df.groupby('LibSize'):
#         real_df_iqr = real_df_libsize.groupby('relation').apply(filter_ptile).reset_index(drop=True)
#         real_df_iqr = real_df_iqr.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
#         real_df_iqr = real_df_iqr[real_df_iqr['relation'] == target_relation].copy()
#
#         surr_df_libsize = surr_df[surr_df['LibSize'] == libsize].copy()
#         surr_df_iqr = surr_df_libsize.groupby('relation_s').apply(filter_ptile).reset_index(drop=True)
#         surr_df_iqr = surr_df_iqr.groupby('relation_s').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
#         surr_df_iqr = surr_df_iqr[surr_df_iqr['relation'] == target_relation].copy()
#         for rel_s, surr_df_iqr_rel in surr_df_iqr.groupby('relation_s'):
#             delta_df_tmp = real_df_iqr.copy()
#             delta_df_tmp['delta_rs_rho'] = delta_df_tmp['rho'] - surr_df_iqr_rel['rho'].values
#             delta_df_tmp['relation_s'] = rel_s
#             delta_df_tmp['rho_s'] = surr_df_iqr_rel['rho'].values
#             delta_df_tmp['surr_var'] = surr_df_iqr_rel['surr_var'].values
#             delta_df_tmp.rename(columns={'rho': 'rho_r'}, inplace=True)
#             delta_df_tmp['delta_r_rho'] = delta_df_tmp['rho_r'] - real_df_min['rho'].values
#             delta_df_tmp['min_rho'] = real_df_min['rho'].values
#             delta_rho_dfs.append(delta_df_tmp)
#
#     delta_rho_df = pd.concat(delta_rho_dfs).reset_index(drop=True)
#     return delta_rho_df
#
#
# from scipy.stats import permutation_test
# def perform_permutation_test(group, n_permutations=10000):
#     # Define the test statistic as the difference in means
#     def statistic(x, y):
#         return np.mean(x) - np.mean(y)
#
#     # Perform the permutation test
#     result = permutation_test((group['rho_r'], group['rho_s']), statistic, alternative='greater',
#                               n_resamples=n_permutations)
#
#     return result.pvalue
#
# def bootstrap_ci(data, n_resamples=10000, confidence_level=0.95):
#     """Bootstrap confidence interval for the mean of a data series."""
#     means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
#     lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
#     upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
#     return lower_bound, upper_bound
#
#
# # Function to perform statistical tests
# def perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='rho_s', confidence_level=0.95):
#     """Performs statistical tests and returns a summary dataframe."""
#     results = []
#     n_permutations = 10000
#     range_label = delta_rho_df['range_label'].iloc[0]
#     if var_2 == 'rho_s':
#         delta_var = 'delta_rs_rho'
#         condition = 'delta_rs'
#         rel_var = 'relation_s'
#     elif var_2 == 'min_rho':
#         delta_var = 'delta_r_rho'
#         condition = 'delta_rho_pos'
#         rel_var = 'relation'
#
#     for relation_s, group in delta_rho_df.groupby(rel_var):
#         if rel_var == 'relation_s':
#             try:
#                 surr_var = group['surr_var'].iloc[0]
#             except Exception as e:
#                 print(f'Error getting surr_var: {e}', file=sys.stderr)
#                 surr_var = relation_s
#         else:
#             surr_var = 'neither'
#
#         # Permutation test for the difference in means
#         p_value_permutation = perform_permutation_test(group, n_permutations)
#
#         # Bootstrap confidence interval for delta_rs_rho
#         ci_low, ci_high = bootstrap_ci(group[delta_var], n_resamples=n_permutations)
#
#         # One-sample t-test on delta_rs_rho
#         t_stat_one, p_value_one = stats.ttest_1samp(group[delta_var], popmean=0, nan_policy='omit')
#
#         # Paired t-test on rho_r and rho_s
#         t_stat_paired, p_value_paired = stats.ttest_rel(
#             group[var_1], group[var_2], alternative='greater', nan_policy='omit'
#         )
#
#         # Confidence interval for the mean delta_rs_rho
#         mean_delta_rho = group[delta_var].mean()
#         # ci_low, ci_high = stats.t.interval(
#         #     confidence_level, len(group) - 1, loc=mean_delta_rho, scale=stats.sem(group['delta_rs_rho'])
#         # )
#
#         # Percentage of positive delta_rs_rho values
#         perc_positive = (group[delta_var] > 0).mean() * 100
#
#         # Append results
#         results.append({
#             'condition': condition,
#             'relation': relation_s,
#             'surr_var': surr_var,
#             'mean_delta_rho': mean_delta_rho,
#             'ci_lower': ci_low,
#             'ci_upper': ci_high,
#             'p_value_permutation': p_value_permutation,
#             'n': 10000,
#             'p_value_one_sample': p_value_one,
#             'p_value_paired': p_value_paired,
#             'perc_positive': perc_positive,
#             'range_label': range_label
#         })
#
#     return results
#
# # Function to fetch and prepare both real and surrogate data
# def fetch_and_prepare_data(real_dfs_references, surr_dfs_references, config, conv_match_d, sample_size=500):
#     meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
#                       'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
#                       'col_var', 'surr_var', 'col_var_id', 'target_var_id']
#
#     max_libsize = 350
#     knn = config.knn_value  # Example attribute for knn
#     target_relation = config.get_nested_attribute('calc_criteria_rates2.target_relation')
#     range_label = '{}_{}'.format(pctile_range[0], pctile_range[1])
#     range_label = range_label.replace('p', '')
#
#     real_df_full = get_real_data(real_dfs_references, meta_variables, max_libsize, knn, sample_size)
#     surr_df_full = get_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, target_relation, sample_size)
#
#     delta_rho_df = construct_delta_rho_df(real_df_full, surr_df_full, target_relation, conv_match_d, sample_size, pctile_range)
#     delta_rho_df['range_label'] = range_label
#     return delta_rho_df
#
#
# ############ Plotting
# def generate_figure_layout(has_surrogates):
#     """Generates the figure layout and axes based on the presence of surrogate data."""
#     if has_surrogates:
#         ncols = 5
#         figsize = (12, 4)
#         width_ratios = [1, 0.2, 0.4, 0.2, 0.4]
#     else:
#         ncols = 3
#         figsize = (8, 3)
#         width_ratios = [1, 0, 1]
#
#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(1, ncols, width_ratios=width_ratios, wspace=0.05)
#     axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
#     if len(axs)==5:
#         iks = [1, 3]
#     elif len(axs)==3:
#         iks = [1]
#     for ik in iks:
#         axs[ik].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
#         axs[ik].set_xticks([])
#         axs[ik].set_yticks([])
#         axs[ik].set_frame_on(False)
#         axs[ik].set_xticklabels([])
#         axs[ik].set_yticklabels([])
#
#     return fig, axs
#
#
# def plot_primary(ax, data, palette, scatter=False):
#     """Plots the primary line and optional scatter plot for real vs. surrogate data."""
#     metric = 'rho'
#     if scatter:
#         sns.scatterplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
#                         alpha=0.2, ax=ax, palette=palette, legend=False)
#
#     sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
#                  legend=False, n_boot=2000, errorbar=("ci", 95), ax=ax, palette=palette)
#     sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
#                  legend=True, n_boot=2000, errorbar=("pi", 50), ax=ax, palette=palette)
#     ax.set_ylabel(r'$\rho$')
#     ax.spines[['top', 'right']].set_visible(False)
#     adjust_left_spine(ax)
#
#
# def plot_density(ax, data, palette):
#     """Plots the KDE density plot for delta rho values."""
#     sns.kdeplot(data=data, y='delta_r_rho', hue='relation', fill=True, common_norm=False,
#                 palette=palette, ax=ax, alpha=0.5, linewidth=1)
#     ax.axhline(0, color='black', linestyle='--', alpha=0.2)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.set_ylabel(r'$\rho_{\text{convergence}} - \rho_{\text{min LibSize}}$')
#
#     adjust_left_spine(ax)
#
#
# def plot_comparison_density(ax, data, palette):
#     """Plots the comparison density between real and surrogate data."""
#     sns.kdeplot(data=data, y='delta_rs_rho', hue='relation_s', fill=True, common_norm=False,
#                 palette=palette, ax=ax, alpha=0.5, linewidth=1)
#     ax.axhline(0, color='black', linestyle='--', alpha=0.2)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.set_ylabel(r'$\rho_{\text{real}} - \rho_{\text{surr}}$')
#     adjust_left_spine(ax)
#
#
# def add_testing_interval_annotation(ax, interval_min, interval_max, ylim):
#     """Adds a rectangle to highlight the testing interval on the plot."""
#     ax.add_patch(plt.Rectangle((interval_min, ylim[0]),
#                                interval_max - interval_min,
#                                ylim[1] - ylim[0],
#                                fill=True, color='gray', alpha=0.1, zorder=0))
#
#     annotation1 = ['Testing Interval: \n{}-{}'.format(int(interval_min),
#                                                       int(interval_max))]
#
#     frac_covered = interval_max/350
#     if frac_covered < .5:
#         _annotation_xloc = .6
#     else:
#         _annotation_xloc = .1
#     block_annotation = ax.annotate('\n'.join(annotation1), xy=(_annotation_xloc, .9),
#                                     # xy=(annotation_xloc, annotation_yloc), #
#                                     xycoords='axes fraction',
#                                     fontsize=10, ha='left', va='center', color='black', alpha=.8)
#
#
# def synchronize_axes(axes):
#     """Synchronizes y-axis limits across multiple axes."""
#     shared_ylims = [min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)]
#     for ax in axes:
#         ax.set_ylim(shared_ylims)
#
#
# def adjust_left_spine(ax):
#     """Adjusts the left spine to extend from 0 to the topmost tick and removes negative ticks and labels."""
#     yticks = ax.get_yticks()
#     yticks = yticks[yticks >= 0]  # Keep only non-negative ticks
#     ax.set_yticks(yticks)
#     ax.spines['left'].set_bounds(yticks[0], yticks[-1])
#
#
# def save_annotated_fig(fig, ax, annotations, positions, save_path_template):
#     """Saves figures with annotated labels if required."""
#     for letter, pos in zip(['a', 'b', 'c'], positions):
#         ax.text(*pos, f'({letter})', transform=ax.transAxes, size=16)
#         plt.savefig(save_path_template.format(letter=letter), bbox_inches='tight')
#         ax.texts.pop()  # Remove text for the next iteration
#
#
# def process_group_workflow(arg_tuple):
#     (grp_d, ind, real_dfs_references, surr_dfs_references, conv_match_d, pctile_range,
#      override, write,  config, calc_location, raw_fig_dir, delta_rho_dir_csv_parts, function_flag) = arg_tuple
#
#     meta_variables = ['tau', 'E', 'train_ind_i', 'knn',
#                       'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
#                       'col_var', 'surr_var', 'col_var_id', 'target_var_id']
#     print('E', grp_d['E'], 'tau', grp_d['tau'], 'ind', ind, 'group_id', grp_d['group_id'], 'function_flag', function_flag)
#
#
#     f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}_Tp{grp_d["Tp"]}_startind{grp_d["train_ind_i"]}_knn{grp_d["knn"]}_lag{grp_d["lag"]}_{grp_d["group_id"]}.pdf'
#     meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
#                       'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
#                       'col_var', 'surr_var', 'col_var_id', 'target_var_id']
#
#     if 'convergence_interval' in conv_match_d:
#         conv_match_d['convergence_interval'] = ast.literal_eval(conv_match_d['convergence_interval'])
#
#     max_libsize = 350
#     knn = 20
#     target_relation = config.calc_convergence_reg.target_relation#get_nested_attribute('calc_criteria_rates2.target_relation')
#     weighted_flag = get_weighted_flag(grp_d)
#
#     range_label = '{}_{}'.format(pctile_range[0], pctile_range[1])
#     real_df_full = get_real_data(real_dfs_references, meta_variables, max_libsize, knn)
#     if real_df_full is None:
#         return None
#     surr_df_full = get_surrogate_data(surr_dfs_references, meta_variables, max_libsize, knn, target_relation)
#     if surr_df_full is None:
#         return None
#     # else:
#     #     print('surr_df_full', len(surr_df_full), type(surr_df_full[0]))
#     #     surr_df_full = pd.concat(surr_df_full).reset_index(drop=True)
#
#     delta_rho_df = construct_delta_rho_df(real_df_full, surr_df_full, target_relation, conv_match_d, sample_size=500,
#                                           pctile_range=pctile_range)
#
#     if delta_rho_df is not None and not delta_rho_df.empty:
#         delta_rho_df['range_label'] = range_label
#         test_results = perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='rho_s')
#         test_results += perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='min_rho')
#         test_results = pd.DataFrame(test_results)
#
#         csv_path = delta_rho_dir_csv_parts / f'delta_rho_grp{grp_d["group_id"]}_ind{ind}.csv'
#         if write=='replace' or not csv_path.exists():
#             test_results.to_csv(csv_path, index=False)
#             print(f'Results saved to {csv_path}', file=sys.stdout, flush=True)
#         else:
#             print(f'Skipped saving, file exists: {csv_path}', file=sys.stdout, flush=True)
#
#         # Plotting phase
#         fig, axs = generate_figure_layout(has_surrogates=not surr_df_full.empty)
#         palette = config.pal.to_dict()
#
#         plot_primary(axs[0], pd.concat([real_df_full, surr_df_full.rename(columns={'relation':'relation2', 'relation_s':'relation'}).drop(columns=['relation2'])]), palette)
#         plot_density(axs[2], delta_rho_df, palette)
#         if not surr_df_full.empty:
#             plot_comparison_density(axs[-1], delta_rho_df, palette)
#
#         add_testing_interval_annotation(axs[0], conv_match_d['convergence_interval'][0],
#                                         conv_match_d['convergence_interval'][1],
#                                         axs[0].get_ylim())
#         synchronize_axes([axs[0], axs[2]] + ([axs[-1]] if not surr_df_full.empty else []))
#
#         plt.tight_layout()
#         axs[0].legend().remove()
#         axs[-1].legend().remove()
#         axs[2].legend().remove()
#
#         fig_sub_dir = raw_fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'
#         fig_sub_dir.mkdir(exist_ok=True, parents=True)
#         fig_target_path = fig_sub_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}_Tp{grp_d["Tp"]}_startind{grp_d["train_ind_i"]}_knn{grp_d["knn"]}_lag{grp_d["lag"]}_{grp_d["group_id"]}.pdf'
#
#         plt.savefig(fig_target_path, bbox_inches='tight')
#         print(f'Figure saved to {fig_target_path}', file=sys.stdout, flush=True)
#
#     return test_results


# Example of how the refactored plotting function would be used:
# fig = process_group_plotting(real_df_full_scatter, surr_df_full_scatter, config, grp_d, conv_match_d, pal, annotate=True)
# plt.savefig('path/to/figure.png')
import pandas as pd
import os

def reformat_delta_rho_file(file_path, grp_id):
    # Extract group_id from the file name (assuming it's part of the file name)
    # group_id = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]

    # Read the delta_rho file
    df = pd.read_csv(file_path)

    # Initialize a dictionary to hold reformatted data
    reformatted_data = {'group_id': grp_id}

    # Populate the dictionary with relevant data from each surr_var
    for surr_var, prefix in [('TSI', 'deltarho_rs_TSI'), ('temp', 'deltarho_rs_temp'),
                             ('neither', 'deltarho_r'), ('neither', 'deltarho_r_final'), ('neither', 'deltarho_r_top')]:
        sub_df = df[df['surr_var'] == surr_var]
        if not sub_df.empty:
            for condition, condition_df in sub_df.groupby('condition'):#sub_df.condition.unique()) > 1:
                # condition = condition.strip('_pos')
                suffix = ''
                if 'rs' not in condition:
                    if 'final' in condition:
                        prefix = 'deltarho_r_final'
                    else:
                        prefix = 'deltarho_r'

                    if '_top' in condition:
                        suffix = '_top'
                    # condition = condition.replace('delta_rho', 'delta_rho_pos')
                    # condition = condition.replace('delta_rho_pos_final_pos', 'delta_rho_pos_final')
                    # suffix = condition.split('delta_rho')[-1]
                    # prefix = f'deltarho_r{suffix}'
                print(prefix+suffix)
                reformatted_data[prefix+suffix] = condition_df['mean_delta_rho'].values[0]
                key = f'perc_pos_{prefix.split("deltarho_", 1)[1]}'+suffix
                # if surr_var != 'neither':
                #     key += f'_{surr_var}'
                print(key)
                reformatted_data[key] = condition_df['perc_positive'].values[0]

    return reformatted_data


import ast
if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    override = args.override  # if args.override not in [None else False
    write_flag = args.write
    if override is False:
        if write_flag in ['append', None]:
            write_flag = 'append'

    second_suffix = ''

    flags = []
    function_flag = 'binding'
    res_flag = ''
    percent_threshold = .01
    if args.flags is not None:
        flags = args.flags
        if 'binding' in flags:
            function_flag = 'binding'

        for flag in args.flags:
            if 'coarse' in flag:
                res_flag = '_' + flag

        numeric_flags = [is_float(val) for val in args.flags if is_float(val) is not None]
        if len(numeric_flags) > 0:
            percent_threshold = numeric_flags[0]

    percent_threshold_label = str(percent_threshold * 100).lstrip('.0').replace('.', 'p')
    if '.' in percent_threshold_label:
        percent_threshold_label = '_' + percent_threshold_label.replace('.', 'p')

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    calc_metric_dir_name = carc_config_d.dirs.calc_metrics_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  # 'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    if args.group_file is not None:
        grp_csv = f'{args.group_file}.csv'
    else:
        grp_csv = f'{carc_config_d.csvs.calc_grp_run_csv}.csv'

    calc_grps_path = calc_carc_mirrored / grp_csv

    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()

    calc_convergence_dir_name = carc_config_d.dirs.calc_convergence_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'
    calc_convergence_dir = calc_location / calc_convergence_dir_name / f'{function_flag}{res_flag}{second_suffix}'
    calc_convergence_dir_csvs = calc_convergence_dir / config.calc_convergence_dir.dirs.csvs
    if 'approach2' in flags:
        perc_threshold_dir = calc_convergence_dir / f'{percent_threshold_label}' / 'approach2'
    else:
        perc_threshold_dir = calc_convergence_dir / f'{percent_threshold_label}'

    if isinstance(args.dir, str):
        if len(args.dir) > 0:
            perc_threshold_dir = perc_threshold_dir/args.dir


    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}' if args.test is not None else ''

    raw_fig_dir = perc_threshold_dir / (config.calc_carc_dir.dirs.ccm_surr_plots_dir_raw + second_suffix)
    raw_fig_dir.mkdir(exist_ok=True, parents=True)
    convergence_grps_df = None

    delta_rho_dir = perc_threshold_dir / config.calc_convergence_dir.dirs.delta_rho
    delta_rho_dir.mkdir(exist_ok=True, parents=True)

    delta_rho_dir_csv_parts = delta_rho_dir / config.delta_rho_dir.dirs.summary_frags
    delta_rho_dir_csv_parts.mkdir(exist_ok=True, parents=True)
    delta_rho_csv_name = f'{config.calc_convergence_dir.csvs.delta_rho_csv}.csv'
    delta_rho_csv = perc_threshold_dir / delta_rho_csv_name
    output_file = perc_threshold_dir / 'convergence_rs_merged.csv'

    convergence_grps_df = pd.read_csv(perc_threshold_dir / 'convergence_grps.csv', index_col=0)

    if 'Tp_tau' in flags:
        rows = []
        # calc_grps_df = calc_grps_df[calc_grps_df['Tp'] !=20].copy()
        for row in calc_grps_df.iterrows():
            row = row[1].to_dict()
            tp_vals = [n * row['tau'] for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            if row['Tp'] in tp_vals:
                rows.append(row)
        calc_grps_df = pd.DataFrame(rows, columns=calc_grps_df.columns)
        print(calc_grps_df)
        raw_fig_dir = perc_threshold_dir / config.calc_carc_dir.dirs.ccm_surr_plots_dir_raw / 'Tp_tau'

    # if Path('/Users/jlanders').exists():
    #
    #     # calc_grps_df = calc_grps_df[(calc_grps_df['tau'] == 4) &
    #     #                             (calc_grps_df['col_var_id'] == 'essel') &
    #     #                             (calc_grps_df['target_var_id'] == 'vieira') &
    #     #                             (calc_grps_df['weighted'] == False)].copy()
    #     calc_grps_df = calc_grps_df[(calc_grps_df['train_ind_i'] == 0) &
    #                                 (calc_grps_df['lag'] == 0) &
    #                                 (calc_grps_df['knn'] == 20) &
    #                                 (calc_grps_df['col_var_id'] == 'essel') &
    #                                 (calc_grps_df['target_var_id'] == 'vieira') &
    #                                 (calc_grps_df['E'] == 4) &
    #                                 (calc_grps_df['tau'] < 8)].copy()
    calc_grps = calc_grps_df.to_dict(orient='records')

    arg_tuples = []
    ind = 0
    convergence_rows = []
    missing_rows = []
    frag_files = os.listdir(delta_rho_dir_csv_parts)
    for grp_d in calc_grps:
        conv_match = convergence_grps_df[convergence_grps_df['group_id'] == grp_d['group_id']].copy()
        if len(conv_match) > 0:
            conv_match_d = conv_match.iloc[0].to_dict()
            if 'convergence_interval' in conv_match_d:
                if type(conv_match_d['convergence_interval'])==str:
                    print('conv_interval', conv_match_d['convergence_interval'])
                    convergence_interval = ast.literal_eval(conv_match_d['convergence_interval'])
                    print(convergence_interval)
                    int_min = int(convergence_interval[0])
                    int_max = int(convergence_interval[1])
        else:
            conv_match_d = {}

        csv_files = [frag_file for frag_file in frag_files if f'grp{grp_d["group_id"]}' in frag_file]
        # print('csv_file', csv_file)
        test_df =None
        if len(csv_files) > 1:
            for csv_file in csv_files:
                csv_path = delta_rho_dir_csv_parts / csv_file
                tmp_test_df = reformat_delta_rho_file(csv_path, grp_d['group_id'])
                interval_min = int(conv_match_d['convergence_interval'][0])
                interval_max = int(conv_match_d['convergence_interval'][1])
                if interval_min == int_min and interval_max == int_max:
                    test_df = tmp_test_df
        elif len(csv_files) == 1:
            csv_path = delta_rho_dir_csv_parts / csv_files[0]
            test_df = reformat_delta_rho_file(csv_path, grp_d['group_id'])
        if test_df is not None:
            convergence_rows.append(test_df)

    convergence_rows_df = pd.DataFrame(convergence_rows)
    convergence_grps_df['group_id'] = convergence_grps_df['group_id'].astype(str)
    convergence_rows_df['group_id'] = convergence_rows_df['group_id'].astype(str)
    master_df = convergence_grps_df.merge(convergence_rows_df, how='left', on='group_id')

    master_df.to_csv(output_file, index=False)



