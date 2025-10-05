from pathlib import Path
from multiprocessing import Pool
import os, sys
import ast

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import permutation_test

from utils.arg_parser import get_parser, parse_flags
from utils.location_helpers import construct_convergence_name
from utils.config_parser import load_config
from utils.data_access import get_real_output, streamline_cause, collect_raw_output, get_weighted_flag, set_df_weighted, write_query_string, relationship_filter, rel_reformat, get_surrogate_output
from utils.location_helpers import *
from utils.run_tools import decide_file_handling


# from utils.data_processing import relationship_filter, rel_reformat
#####
# Functions to construct figures underpinnig summary grid results (seen in SI Fig 3, details)
#####

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

############ Data Grab
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

import re
def remove_numbers(input_string):
    # Use regex to remove all digits from the string
    return re.sub(r'\d+', '', input_string)


# # Function to fetch and prepare surrogate data
# def get_surrogate_data(surr_dfs_references, grp_path, meta_variables, max_libsize, knn, _rel,
#                        sample_size=400):
#     surr_dfs = []
#     ctr = 0
#     for df_csv_name in surr_dfs_references:
#         surr_df_full = pd.read_csv(grp_path / df_csv_name)
#         surr_df = relationship_filter(surr_df_full, _rel) #surr_df_full[surr_df_full['relation'].isin([_rel, _rel.replace('influences', 'causes')])].copy()
#
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

# IQR filtering function


def filter_ptile(group, l=0.25, u=0.75):
    Q1 = group['rho'].quantile(l)
    Q3 = group['rho'].quantile(u)
    return group[(group['rho'] >= Q1) & (group['rho'] <= Q3)]

filter_top_ptile = lambda x: filter_ptile(x, l=0.9, u=1)



# Function to construct delta rho dataframe
def construct_delta_rho_df(real_df_full, surr_df_full, target_relation, conv_match_d, sample_size=500, pctile_range=[.25, .75]):
    if 'convergence_interval' not in conv_match_d:
        print('No convergence interval found', file=sys.stderr, flush=True)
        return None
    print('construct_delta_rho_df', file=sys.stderr, flush=True)
    if isinstance(conv_match_d['convergence_interval'], tuple) == True:
        (interval_min, interval_max) = conv_match_d['convergence_interval']
    else:
        print('conv_interval?', conv_match_d['convergence_interval'], type(conv_match_d['convergence_interval']), file=sys.stderr, flush=True)
        (interval_min, interval_max) = ast.literal_eval(conv_match_d['convergence_interval'])
    # try:
    #     interval_min, interval_max = conv_match_d['convergence_interval']
    # except:
    #     (interval_min, interval_max) = conv_match_d['convergence_interval']
    # else:
    #     print('Error unpacking convergence interval', file=sys.stderr, flush=True)
    #     return None
    real_df = real_df_full[(real_df_full['LibSize'] >= interval_min) & (real_df_full['LibSize'] <= interval_max)].copy()
    surr_df = surr_df_full[(surr_df_full['LibSize'] >= interval_min) & (surr_df_full['LibSize'] <= interval_max)].copy()

    real_df_min = relationship_filter(real_df_full[(real_df_full['LibSize'] < 40)].copy(), target_relation)
    real_df_min['rho'] = real_df_min['rho'].astype(float)
    real_df_min = real_df_min.groupby('LibSize').apply(filter_ptile).reset_index(drop=True)
    real_df_min = real_df_min.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)

    real_df_min_top = relationship_filter(real_df_full[(real_df_full['LibSize'] < 40)].copy(), target_relation)
    real_df_min_top['rho'] = real_df_min_top['rho'].astype(float)
    real_df_min_top = real_df_min_top.groupby('LibSize').apply(filter_top_ptile).reset_index(drop=True)
    real_df_min_top = real_df_min_top.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(
        drop=True)

    real_df_final = real_df_full[(real_df_full['LibSize'] > 300)].copy()
    real_df_final['rho'] = real_df_final['rho'].astype(float)
    real_df_final = real_df_final.groupby('LibSize').apply(filter_ptile).reset_index(drop=True)
    real_df_final = real_df_final.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(
        drop=True)
    real_df_final = relationship_filter(real_df_final, target_relation)#[real_df_final['relation'] == target_relation].copy()

    delta_rho_dfs = []
    for libsize, real_df_libsize in real_df.groupby('LibSize'):
        real_df_libsize['rho'] = real_df_libsize['rho'].astype(float)
        real_df_iqr = real_df_libsize.groupby('relation').apply(filter_ptile).reset_index(drop=True)
        real_df_iqr = real_df_iqr.groupby('relation').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
        real_df_iqr = relationship_filter(real_df_iqr, target_relation)#[real_df_iqr['relation'] == target_relation].copy()

        real_df_final_top = real_df_libsize.groupby('relation').apply(filter_top_ptile).reset_index(drop=True)
        real_df_final_top = real_df_final_top.groupby('relation').apply(
            lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
        real_df_final_top = relationship_filter(real_df_final_top, target_relation)#[real_df_final_top['relation'] == target_relation].copy()

        surr_df_libsize = surr_df[surr_df['LibSize'] == libsize].copy()
        surr_df_libsize['rho'] = surr_df_libsize['rho'].astype(float)
        surr_df_iqr = surr_df_libsize.groupby('relation_s').apply(filter_ptile).reset_index(drop=True)
        surr_df_iqr = surr_df_iqr.groupby('relation_s').apply(lambda x: x.sample(n=sample_size, replace=True)).reset_index(drop=True)
        surr_df_iqr = relationship_filter(surr_df_iqr, target_relation)#[surr_df_iqr['relation'] == target_relation].copy()
        for rel_s, surr_df_iqr_rel in surr_df_iqr.groupby('relation_s'):
            delta_df_tmp = real_df_iqr.copy()
            delta_df_tmp['delta_rs_rho'] = delta_df_tmp['rho'] - surr_df_iqr_rel['rho'].values
            delta_df_tmp['relation_s'] = rel_s
            delta_df_tmp['rho_s'] = surr_df_iqr_rel['rho'].values
            delta_df_tmp['surr_var'] = surr_df_iqr_rel['surr_var'].values
            delta_df_tmp.rename(columns={'rho': 'rho_r'}, inplace=True)
            delta_df_tmp['delta_r_rho'] = delta_df_tmp['rho_r'] - real_df_min['rho'].values
            delta_df_tmp['min_rho'] = real_df_min['rho'].values
            delta_df_tmp['min_rho_top'] = real_df_min_top['rho'].values

            delta_df_tmp['final_rho'] = real_df_final['rho'].values
            delta_df_tmp['final_rho_top'] = real_df_final_top['rho'].values

            delta_df_tmp['delta_r_rho_final'] = real_df_final['rho'].values - real_df_min['rho'].values
            delta_df_tmp['delta_r_rho_top'] = real_df_final_top['rho'].values - real_df_min_top['rho'].values
            delta_rho_dfs.append(delta_df_tmp)

    delta_rho_df = pd.concat(delta_rho_dfs).reset_index(drop=True)
    print('delta_rho_df', len(delta_rho_df), file=sys.stderr, flush=True)
    return delta_rho_df


# Statistical Testing
def perform_permutation_test(group, n_permutations=10000):
    # Define the test statistic as the difference in means
    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    # Perform the permutation test
    result = permutation_test((group['rho_r'], group['rho_s']), statistic, alternative='greater',
                              n_resamples=n_permutations)

    return result.pvalue

def bootstrap_ci(data, n_resamples=10000, confidence_level=0.95):
    """Bootstrap confidence interval for the mean of a data series."""
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
    return lower_bound, upper_bound


# Function to perform statistical tests
def perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='rho_s', confidence_level=0.95):
    """Performs statistical tests and returns a summary dataframe."""
    results = []
    n_permutations = 10000
    range_label = delta_rho_df['range_label'].iloc[0]
    if var_2 == 'rho_s':
        delta_var = 'delta_rs_rho'
        condition = 'delta_rs'
        rel_var = 'relation_s'
    elif var_2 == 'min_rho':
        delta_var = 'delta_r_rho'
        condition = 'delta_rho_pos'
        rel_var = 'relation'
        if var_1 == 'final_rho':
            delta_var = 'delta_r_rho_final'
            condition = 'delta_rho_final_pos'
            rel_var = 'relation'
    elif var_2 == 'min_rho_top':
        var_1 = 'final_rho_top'
        delta_var = 'delta_r_rho_top'
        condition = 'delta_r_rho_top_pos'
        rel_var = 'relation'

    for relation_s, group in delta_rho_df.groupby(rel_var):
        if rel_var == 'relation_s':
            try:
                surr_var = group['surr_var'].iloc[0]
            except Exception as e:
                print(f'Error getting surr_var: {e}', file=sys.stderr)
                surr_var = relation_s
        else:
            surr_var = 'neither'

        # Permutation test for the difference in means
        p_value_permutation = perform_permutation_test(group, n_permutations)

        # Bootstrap confidence interval for delta_rs_rho
        ci_low, ci_high = bootstrap_ci(group[delta_var], n_resamples=n_permutations)

        # One-sample t-test on delta_rs_rho
        t_stat_one, p_value_one = stats.ttest_1samp(group[delta_var], popmean=0, nan_policy='omit')

        # Paired t-test on rho_r and rho_s
        t_stat_paired, p_value_paired = stats.ttest_rel(
            group[var_1], group[var_2], alternative='greater', nan_policy='omit')

        # mean delta_rs_rho
        mean_delta_rho = group[delta_var].mean()

        # Percentage of positive delta_rs_rho values
        perc_positive = (group[delta_var] > 0).mean() * 100

        # Append results
        results.append({
            'condition': condition,
            'relation': relation_s,
            'surr_var': surr_var,
            'mean_delta_rho': mean_delta_rho,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'p_value_permutation': p_value_permutation,
            'n': 10000,
            'p_value_one_sample': p_value_one,
            'p_value_paired': p_value_paired,
            'perc_positive': perc_positive,
            'range_label': range_label
        })

    return results

# Function to fetch and prepare both real and surrogate data
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


############ Plotting
### Layout and Axes
def generate_figure_layout(has_surrogates):
    """Generates the figure layout and axes based on the presence of surrogate data."""
    if has_surrogates:
        ncols = 5
        figsize = (8, 3)
        width_ratios = [1, 0.2, 0.4, 0.2, 0.4]
        print('has_surrogates', file=sys.stderr, flush=True)
    else:
        ncols = 3
        figsize = (8, 3)
        width_ratios = [1, 0, 1]
        print('no_surrogates', file=sys.stderr, flush=True)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, ncols, width_ratios=width_ratios, wspace=0.05)
    axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    if len(axs)==5:
        iks = [1, 3]
    elif len(axs)==3:
        iks = [1]
    for ik in iks:
        axs[ik].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        axs[ik].set_xticks([])
        axs[ik].set_yticks([])
        axs[ik].set_frame_on(False)
        axs[ik].set_xticklabels([])
        axs[ik].set_yticklabels([])

    return fig, axs

def synchronize_axes(axes):
    """Synchronizes y-axis limits across multiple axes."""
    shared_ylims = [min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)]
    for ax in axes:
        ax.set_ylim(shared_ylims)

def adjust_left_spine(ax):
    """Adjusts the left spine to extend from 0 to the topmost tick and removes negative ticks and labels."""
    yticks = ax.get_yticks()
    yticks = yticks[yticks >= 0]  # Keep only non-negative ticks
    ax.set_yticks(yticks)
    ax.spines['left'].set_bounds(yticks[0], yticks[-1])


### Plotting Functions
# CCM output: library size (LibSize) vs. correlation (rho) plots
def plot_primary(ax, data_d, palette, scatter=False, lines=False, pi=90, pi_shading=True,
                 n_boot=1000, libsize_max=350, surr_lw=.75, real_lw=3, surr_kwargs={}):
    """Plots the primary line and optional scatter plot for real vs. surrogate data."""
    metric = 'rho'
    # if isinstance(data, tuple):
    #     if len(data) == 2:
    #         data = data[0]

    full = data_d['full']
    data = data_d['data']

    data = data[data['LibSize'] <= libsize_max].copy()
    data = data[data['relation'].isin(palette.keys())].copy()
    if full is not None:
        full = full[full['LibSize'] <= libsize_max].copy()
        full = full[full['relation'].isin(palette.keys())].copy()
    if scatter:
        if full is not None:
            sns.scatterplot(data=full.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                        alpha=0.2, ax=ax, palette=palette, legend=False, s=1)
        else:
            sns.scatterplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                        alpha=0.2, ax=ax, palette=palette, legend=False)

    if lines is True:
        # for surr_num, surr_num_df in data.groupby(['surr_num']):
        #     sns.lineplot(data=surr_num_df.rename(columns={metric: 'rho'}), x='LibSize', y='rho',
        #                  hue='relation', legend=False, ax=ax, palette=palette,alpha=.2)
        sns.lineplot(data=data[data['surr_var']=='neither'].rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                     legend=True, n_boot=n_boot, ax=ax, palette=palette, lw=real_lw)
        sns.lineplot(data=data[data['surr_var']!='neither'].rename(columns={metric: 'rho'}), x='LibSize', y='rho',
                     hue='relation', legend=True, ax=ax, palette=palette,
                     estimator=None, alpha=.2, units='surr_num', lw=surr_lw)
        sns.lineplot(data=data[data['surr_var']=='neither'].rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                     legend=False, n_boot=n_boot, ax=ax, palette=palette, lw=real_lw)

    if pi_shading is True:
        sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                     legend=False, n_boot=n_boot, errorbar=("pi", pi), ax=ax, palette=palette)

    # sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
    #              legend=False, ax=ax, n_boot=n_boot, errorbar=("pi", pi), palette=palette) #n_boot=3000, errorbar=("ci", 95),
    #
    if lines is False:
        # if 'surr_rel' in surr_kwargs.keys():
        #     surr_data = data[data['relation'].isin(surr_kwargs['surr_rel'])].copy()
        #     real_data = data[data['surr_var'] == 'neither'].copy()
        #     data = pd.concat([real_data, surr_data]).copy()

        sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
                 legend=True,  n_boot=n_boot, ax=ax, palette=palette) #
    # sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
    #              legend=False, n_boot=3000, errorbar=("ci", 95), ax=ax, palette=palette)
    # sns.lineplot(data=data.rename(columns={metric: 'rho'}), x='LibSize', y='rho', hue='relation',
    #              legend=True, n_boot=2000, errorbar=("pi", 50), ax=ax, palette=palette)
    ax.set_ylabel(r'$\rho$')
    ax.spines[['top', 'right']].set_visible(False)
    adjust_left_spine(ax)


# (Real) delta rho KDE plots: delta rho (above and below)
def plot_density(ax, data, palette, var='delta_r_rho'):
    """Plots the KDE density plot for delta rho values."""
    sns.kdeplot(data=data, y=var, hue='relation', fill=True, common_norm=False,
                palette=palette, ax=ax, alpha=0.5, linewidth=1)

    if var == 'delta_r_rho':
        sns.kdeplot(data=data, y='delta_r_rho_top', hue='relation', fill=False, common_norm=False,
                    palette=palette, ax=ax, alpha=1, linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    if var == 'delta_r_rho':
        ax.set_ylabel(r'$\rho_{\text{convergence}} - \rho_{\text{min LibSize}}$')
    elif var == 'delta_r_rho_final':
        ax.set_ylabel(r'$\rho_{\text{final LibSize}} - \rho_{\text{min LibSize}}$')

    adjust_left_spine(ax)


# (Real-Surrogate) KDE plots: delta rho (real - surrogate) vs. library size (LibSize)
def plot_comparison_density(ax, data, palette):
    """Plots the comparison density between real and surrogate data."""
    sns.kdeplot(data=data, y='delta_rs_rho', hue='relation_s', fill=True, common_norm=False,
                palette=palette, ax=ax, alpha=0.5, linewidth=1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel(r'$\rho_{\text{real}} - \rho_{\text{surr}}$')
    adjust_left_spine(ax)


# Function to add a rectangle annotation for the testing interval
def add_testing_interval_annotation(ax, interval_min, interval_max, ylim):
    """Adds a rectangle to highlight the testing interval on the plot."""
    ax.add_patch(plt.Rectangle((interval_min, ylim[0]),
                               interval_max - interval_min,
                               ylim[1] - ylim[0],
                               fill=True, color='gray', alpha=0.1, zorder=0))

    annotation1 = ['Testing Interval: \n{}-{}'.format(int(interval_min),
                                                      int(interval_max))]

    frac_covered = interval_max/350
    if frac_covered < .5:
        _annotation_xloc = .6
    else:
        _annotation_xloc = .1
    block_annotation = ax.annotate('\n'.join(annotation1), xy=(_annotation_xloc, .9),
                                    # xy=(annotation_xloc, annotation_yloc), #
                                    xycoords='axes fraction',
                                    fontsize=10, ha='left', va='center', color='black', alpha=.8)


# Helper function to save annotated figures
def save_annotated_fig(fig, ax, annotations, positions, save_path_template):
    """Saves figures with annotated labels if required."""
    for letter, pos in zip(['a', 'b', 'c'], positions):
        ax.text(*pos, f'({letter})', transform=ax.transAxes, size=16)
        plt.savefig(save_path_template.format(letter=letter), bbox_inches='tight')
        ax.texts.pop()  # Remove text for the next iteration


def write_to_file(ccm_out_df, df_path, overwrite=False):
    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len',
                   'train_ind_i']
    ccm_out_df = ccm_out_df[[col for col in ccm_out_df.columns if col not in remove_cols]].copy()

    try:
        ccm_out_df_0 = pd.read_csv(df_path, index_col=0)
        if overwrite == False:
            ccm_out_df_0 = ccm_out_df_0[[col for col in ccm_out_df_0.columns if col not in remove_cols]].copy()
            ccm_out_df = pd.concat([ccm_out_df_0, ccm_out_df])
            ccm_out_df.reset_index(drop=True, inplace=True)
    except:
        pass

    ccm_out_df.to_csv(df_path)
    print('!\twrote to file: ', df_path)

############ Main Processing Function
def process_group_workflow(arg_tuple):
    (grp_d, ind, real_dfs_references, surr_dfs_references, conv_match_d, pctile_range,
     args,  config, calc_location, output_dir, raw_fig_dir, delta_rho_dir_csv_parts, function_flag) = arg_tuple
    print('E', grp_d['E'], 'tau', grp_d['tau'], 'ind', ind, 'group_id', grp_d['group_id'], 'function_flag', function_flag)
    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']

    grp_path = set_grp_path(output_dir, grp_d)
    # grp_path = output_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'/ f'E{grp_d["E"]}_tau{grp_d["tau"]}'


    if 'convergence_interval' in conv_match_d:
        if type(conv_match_d['convergence_interval']) == str:
            try:
                conv_match_d['convergence_interval'] = ast.literal_eval(conv_match_d['convergence_interval'].replace('np.float64', ''))
            except Exception as e:
                print(f'Error parsing convergence interval: {e}', file=sys.stderr)
            # conv_match_d['convergence_interval'] = [0, 0]

    max_libsize = 325
    knn = 20
    target_relation = config.target_relation#get_nested_attribute('calc_criteria_rates2.target_relation')
    weighted_flag = get_weighted_flag(grp_d)

    range_label = '{}_{}'.format(pctile_range[0], pctile_range[1])

    var = 'delta_r_rho'
    fig_sub_dir = raw_fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / var
    # fig_sub_dir.mkdir(exist_ok=True, parents=True)
    fig_target_path = fig_sub_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}_Tp{grp_d["Tp"]}_startind{grp_d["train_ind_i"]}_knn{grp_d["knn"]}_lag{grp_d["lag"]}_{grp_d["group_id"]}.pdf'
    # if fig_target_path.exists()==True:
    #     if override == False:
    #         print(f'Skipped processing, file exists: {fig_target_path}', file=sys.stdout, flush=True)
    #         return None
    fig_existence = fig_target_path.exists()
    run_continue, overwrite = decide_file_handling(args, fig_existence)
    if run_continue == False:
        print(f'\Skipped processing, file exists: {fig_target_path}', file=sys.stdout, flush=True)
        return None

    print('fig_target_path', fig_target_path, file=sys.stdout, flush=True)
    ik = 0
    while ik < 10:
        real_df_full = get_real_output(real_dfs_references, grp_path, meta_variables, max_libsize, knn)
        if real_df_full is None:
            print('No real data found', grp_d, file=sys.stderr, flush=True)
            if ik == 9:
                return None
            else:
                ik += 1
        else:
            ik = 10
    print('real_df_full', len(real_df_full), file=sys.stderr, flush=True)

    ik = 0
    print('surr_dfs_references', surr_dfs_references, file=sys.stdout, flush=True)
    while ik<10:
        surr_df_full = get_surrogate_output(surr_dfs_references, grp_path, meta_variables, max_libsize, knn, target_relation)
        if surr_df_full is None:
            print('No surrogate data found', grp_d, file=sys.stderr, flush=True)
            if ik == 9:
                return None
            else:
                ik +=1
        else:
            ik = 10
    print('surr_df_full', len(surr_df_full), file=sys.stderr, flush=True)

    ik = 0
    while ik < 10:
        delta_rho_df = construct_delta_rho_df(real_df_full, surr_df_full, target_relation, conv_match_d, sample_size=500,
                                          pctile_range=pctile_range)
        if delta_rho_df is None:
            print('No delta rho data found', grp_d, file=sys.stderr, flush=True)
            ik += 1
        else:
            ik = 10
    print('delta_rho_df', len(delta_rho_df), file=sys.stderr, flush=True)


    if delta_rho_df is not None and not delta_rho_df.empty:
        csv_path = delta_rho_dir_csv_parts / f'delta_rho_grp{grp_d["group_id"]}_ind{ind}.csv'
        existence = csv_path.exists()
        run_continue, overwrite = decide_file_handling(args, existence)
        if run_continue == False:
            print(f'Skipped processing, file exists: {csv_path}', file=sys.stdout, flush=True)
        else:
            delta_rho_df['range_label'] = range_label
            test_results = perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='rho_s')
            test_results += perform_statistical_tests(delta_rho_df, var_1='rho_r', var_2='min_rho')
            test_results += perform_statistical_tests(delta_rho_df, var_1='final_rho', var_2='min_rho')
            test_results += perform_statistical_tests(delta_rho_df, var_1='final_rho_top', var_2='min_rho_top')

            test_results = pd.DataFrame(test_results)
            write_to_file(test_results, csv_path, overwrite=overwrite)
            print(f'Results saved to {csv_path}', file=sys.stdout, flush=True)
        #
        #
        #
        # if write=='replace' or not csv_path.exists():
        #     test_results.to_csv(csv_path, index=False)
        #     print(f'Results saved to {csv_path}', file=sys.stdout, flush=True)
        # else:
        #     print(f'Skipped saving, file exists: {csv_path}', file=sys.stdout, flush=True)

        # Plotting phase
        palette = config.pal.to_dict()

        for var in ['delta_r_rho', 'delta_r_rho_final']:
            print('var', var, file=sys.stderr, flush=True)
            fig, axs = generate_figure_layout(has_surrogates=not surr_df_full.empty)

            primary_df = pd.concat([real_df_full, surr_df_full.rename(columns={'relation':'relation2', 'relation_s':'relation'}).drop(columns=['relation2'])])
            primary_df = rel_reformat(rel_reformat(primary_df, 'relation'), 'relation_s')
            plot_primary(axs[0], primary_df, palette)
            plot_density(axs[2], rel_reformat(rel_reformat(delta_rho_df, 'relation'), 'relation_s') , palette, var=var)
            if not surr_df_full.empty:
                plot_comparison_density(axs[-1], rel_reformat(rel_reformat(delta_rho_df, 'relation'), 'relation_s'), palette)
                print('plot_comparison_density', file=sys.stderr, flush=True)

            add_testing_interval_annotation(axs[0], conv_match_d['convergence_interval'][0],
                                            conv_match_d['convergence_interval'][1],
                                            axs[0].get_ylim())
            synchronize_axes([axs[0], axs[2]] + ([axs[-1]] if not surr_df_full.empty else []))

            axs[0].legend().remove()
            axs[-1].legend().remove()
            axs[2].legend().remove()
            plt.tight_layout()

            fig_sub_dir = raw_fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'/var
            fig_sub_dir.mkdir(exist_ok=True, parents=True)
            fig_target_path = fig_sub_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}_Tp{grp_d["Tp"]}_startind{grp_d["train_ind_i"]}_knn{grp_d["knn"]}_lag{grp_d["lag"]}_{grp_d["group_id"]}.pdf'

            plt.savefig(fig_target_path, bbox_inches='tight')
            print(f'Figure saved to {fig_target_path}', file=sys.stdout, flush=True)
            # print(os.listdir(str(fig_sub_dir)), file=sys.stdout, flush=True)
    else:
        print('No delta rho data found, exiting failed', grp_d, file=sys.stderr, flush=True)



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

    # override = args.override  # if args.override not in [None else False
    # write_flag = args.write
    # if override is False:
    #     if write_flag in ['append', None]:
    #         write_flag = 'append'

    second_suffix = ''

    flags = []
    print('args.flags', args.flags, file=sys.stdout, flush=True)
    percent_threshold, function_flag, res_flag, second_suffix = parse_flags(args,
                                                                            default_percent_threshold=.05,
                                                                            default_function_flag='binding',
                                                                            default_res_flag='',
                                                                            default_second_suffix=second_suffix)


    print('second_suffix', second_suffix, file=sys.stdout, flush=True)
    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    # calc_metric_dir_name = carc_config_d.dirs.calc_metrics_dir  # config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_location = set_calc_path(args, proj_dir, config)
    output_dir = set_output_path(args, calc_location, config)

    # Configuration Groups
    if args.group_file is not None:
        grp_csv = f'{args.group_file}.csv'
    else:
        grp_csv = f'{carc_config_d.csvs.calc_grp_run_csv}.csv'

    calc_grps_path = calc_carc_mirrored / grp_csv
    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()

    # Convergence information location
    calc_convergence_dir_parent = calc_location / construct_convergence_name(args, carc_config_d, percent_threshold, '')
    # calc_convergence_dir_csvs = calc_convergence_dir / config.calc_convergence_dir.dirs.csvs

    # # Create directories for saving figures & summary statistics
    # raw_fig_dir = calc_convergence_dir / (config.calc_carc_dir.dirs.ccm_surr_plots_dir_raw + second_suffix)
    # raw_fig_dir.mkdir(exist_ok=True, parents=True)
    # print('raw_fig_dir', raw_fig_dir, file=sys.stdout, flush=True)
    #
    # delta_rho_dir = calc_convergence_dir / (config.calc_convergence_dir.dirs.delta_rho + second_suffix)
    # delta_rho_dir.mkdir(exist_ok=True, parents=True)
    #
    # delta_rho_dir_csv_parts = delta_rho_dir / config.delta_rho_dir.dirs.summary_frags
    # delta_rho_dir_csv_parts.mkdir(exist_ok=True, parents=True)

    # Load convergence groups

    convergence_grps_df = None
    if args.file is not None:
        convergence_file = f'{args.file}.csv'
    else:
        convergence_file = 'convergence_grps.csv'
    convergence_grps_df = pd.read_csv(calc_convergence_dir_parent / convergence_file, index_col=0)

    if 'Tp_tau' in flags:
        rows = []
        # calc_grps_df = calc_grps_df[calc_grps_df['Tp'] !=20].copy()
        for row in calc_grps_df.iterrows():
            row = row[1].to_dict()
            tp_vals = [n * row['tau'] for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            if row['Tp'] in tp_vals:
                rows.append(row)
        calc_grps_df = pd.DataFrame(rows, columns=calc_grps_df.columns)
        raw_fig_dir = calc_convergence_dir / config.calc_carc_dir.dirs.ccm_surr_plots_dir_raw / 'Tp_tau'

    pctile_range = [.25, .75]
    query_keys = config.calc_criteria_rates2.query_keys

    # Test locally
    if Path('/Users/jlanders').exists():

        calc_grps = calc_grps_df.to_dict(orient='records')

        arg_tuples = []
        ind = 0
        for grp_d in calc_grps:

            calc_convergence_dir, calc_convergence_dir_csvs, convergence_dir_csv_parts, fig_dir = set_convergence_paths(calc_convergence_dir_parent, config, grp_d)
            convergence_grps_df = pd.read_csv(calc_convergence_dir / convergence_file, index_col=0)

            conv_match = convergence_grps_df[convergence_grps_df['group_id'] == grp_d['group_id']].copy()
            if len(conv_match) > 0:
                conv_match_d = conv_match.iloc[0].to_dict()
                if len(conv_match_d)>0:
                    grp_path = set_grp_path(output_dir, grp_d)
                    delta_rho_dir, delta_rho_dir_csv_parts = set_delta_rho_paths(calc_convergence_dir, config, grp_d)
                    raw_fig_dir = set_raw_fig_dir(calc_convergence_dir, config, grp_d, second_suffix)

                    files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
                    real_dfs_references = [file for file in files if 'neither' in file]
                    surr_dfs_references = [file for file in files if 'neither' not in file]

                    arg_tuples.append((grp_d, ind, real_dfs_references, surr_dfs_references, conv_match_d, pctile_range,
                                       args, config, calc_location,output_dir,
                                       raw_fig_dir, delta_rho_dir_csv_parts, function_flag))
            else:
                conv_match_d = {}
            ind += 1
        num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([4, len(arg_tuples)])))

        # Use multiprocessing to parallelize the process
        with Pool(num_cpus) as pool:
            pool.map(process_group_workflow, arg_tuples)

    # Run on Cluster
    else:

        if args.inds is not None:
            index = int(args.inds[-1])
        else:
            sys.exit(0)

        grp_ds = calc_grps_df.to_dict(orient='records')
        if index >= len(grp_ds):
            print('index out of range', file=sys.stdout, flush=True)
            sys.exit(1)
        else:
            grp_d = grp_ds[index]

        grp_path = set_grp_path(output_dir, grp_d)
        # grp_path = output_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'E{grp_d["E"]}_tau{grp_d["tau"]}'
        files = [file for file in os.listdir(grp_path) if (file.endswith('.csv'))]
        real_dfs_references = [file for file in files if 'neither' in file]
        surr_dfs_references = [file for file in files if 'neither' not in file]

        calc_convergence_dir, calc_convergence_dir_csvs, convergence_dir_csv_parts, fig_dir = set_convergence_paths(
            calc_convergence_dir_parent, config, grp_d)

        conv_match = convergence_grps_df[convergence_grps_df['group_id'] == grp_d['group_id']].copy()
        if len(conv_match) > 0:
            conv_match_d = conv_match.iloc[0].to_dict()
            print(conv_match_d, file=sys.stdout, flush=True)

            delta_rho_dir, delta_rho_dir_csv_parts = set_delta_rho_paths(calc_convergence_dir, config, grp_d)
            raw_fig_dir = set_raw_fig_dir(calc_convergence_dir, config, grp_d, second_suffix)
            arg_tuple = (grp_d, index, real_dfs_references, surr_dfs_references, conv_match_d, pctile_range, args, config, calc_location,output_dir,
                         raw_fig_dir, delta_rho_dir_csv_parts, function_flag)

            process_group_workflow(arg_tuple)
            print('figures built successfully!', file=sys.stdout, flush=True)
        else:
            conv_match_d = {}
            print('no convergence match', grp_d, file=sys.stdout, flush=True)



