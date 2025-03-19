from multiprocessing import Pool
import os
from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pyleoclim as pyleo

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

# Define directories for notebooks and calculations
notebooks_dir = Path(os.getcwd()) / 'notebooks'
calc_carc = notebooks_dir / 'calc_carc'

calc_metric_dir_name = 'calc_metrics_scaled'
calc_location = notebooks_dir / ('calc_local_tmp' if Path('/Users/jlanders').exists() else 'calc_carc')

pd.option_context('mode.use_inf_as_na', True)

proj_d = {
    'erb': {'tau': [10, 5], 'temp_var': 'erb2022_da_temp', 'E': [6, 9],
            'surr_file_name': 'pe_hol_tsi_temp_surr_vieira_erb.csv'},
    'essel': {'tau': [5, 10], 'temp_var': 'essel2023_opt_temp', 'E': [14],
              'surr_file_name': 'pe_hol_tsi_temp_surr_wu_essel.csv'},
    'vieira': {'tau': [5, 10], 'tsi_var': 'vieira_tsi', 'surr_file_name': 'pe_hol_tsi_temp_surr_vieira_erb.csv'},
    'wu': {'tau': [5, 10], 'tsi_var': 'wu_tsi', 'surr_file_name': 'pe_hol_tsi_temp_surr_wu_essel.csv'}
}


def prepare_data(grp_d):
    """Load and prepare data based on the group details."""
    temp_var = proj_d[grp_d['temp_author']]['temp_var']
    tsi_var = proj_d[grp_d['tsi_author']]['tsi_var']

    data = pd.read_csv(notebooks_dir.parent / 'data' / 'hol_ct_data_short.csv', index_col=0)
    data = data[['time', temp_var, tsi_var]].rename(columns={'time': 'date', temp_var: 'temp', tsi_var: 'TSI'})

    shifted = data.copy()
    shifted[grp_d['target_var']] = shifted[grp_d['target_var']].shift(0)
    shifted = shifted.dropna(subset=[grp_d['target_var'], grp_d['col_var']]).reset_index(drop=True)

    return shifted


def compute_psd(subset, var_name, label_prefix):
    """Compute the power spectral density (PSD) using Welch method."""
    ac_val = pyleo.utils.tsmodel.ar1_fit(subset[var_name])
    series = pyleo.Series(time=subset['date'], value=subset[var_name], value_name=f'{label_prefix}',
                          label=f'{ac_val:.3f}', time_name='years')
    # _, detrending = pyleo.utils.tsutils.detrend(series.value, method='emd')
    detrended_series = pyleo.Series(time=series.time, value=detrending, value_name=f'{label_prefix}-trend',
                                    label=var_name, time_name='years')
    psd_welch = series.spectral(method='welch')

    return series, detrended_series, psd_welch


def get_sig_periods(psd):
    psd_signif = psd.signif_test(number=1000, method='ar1sim')
    freq = psd_signif.frequency
    indexes_of_superior = np.argwhere(psd_signif.amplitude > psd_signif.signif_qs.psd_list[0].amplitude)
    freq_superior = freq[indexes_of_superior]
    return psd_signif, (1 / freq_superior).flatten()

def plot_psd(ax, psd_list, var=None, important_periods=None, plot=True):
    """Plot the PSD using Welch method and mark significant periods with different colors for common and unique periods."""
    period_of_superiors = []
    for psd_welch in psd_list:
        psd_signif, signif_periods = get_sig_periods(psd_welch)
        # psd_welch_signif = psd_welch.signif_test(number=1000, method='ar1sim')
        if plot is True:
            psd_signif.plot(ax=ax, linewidth=2, legend=False)
        # freq = psd_welch_signif.frequency
        # indexes_of_superior = np.argwhere(psd_welch_signif.amplitude > psd_welch_signif.signif_qs.psd_list[0].amplitude)
        # freq_superior = freq[indexes_of_superior]
        period_of_superiors.extend(signif_periods)

    if plot is True:
        ymax = max(ax.get_ylim())
    else:
        ymax=20
    if var is not None:
        if important_periods is None:
            important_periods = {var: period_of_superiors}
        else:
            important_periods[var] = period_of_superiors

    ax.set_ylim([10, ymax])
    ax.yaxis.tick_right()
    ax.set_ylabel('PSD using Welch', labelpad=15, rotation=270)
    ax.yaxis.set_label_position("right")
    ax.spines['right'].set_position(('outward', 10))
    ax.spines['right'].set_visible(True)  # Ensure the right spine is visible
    ax.spines['left'].set_visible(False)

    return important_periods, ax

#
# def plot_psd(ax, psd_list, var=None, important_periods=None):
#     """Plot the PSD using Welch method and mark significant periods with different colors for common and unique periods."""
#     period_of_superiors = []
#     for psd_welch in psd_list:
#         psd_welch_signif = psd_welch.signif_test(number=200, method='ar1sim')
#         psd_welch_signif.plot(ax=ax, linewidth=2, legend=False)
#         freq = psd_welch_signif.frequency
#         indexes_of_superior = np.argwhere(psd_welch_signif.amplitude > psd_welch_signif.signif_qs.psd_list[0].amplitude)
#         freq_superior = freq[indexes_of_superior]
#         period_of_superiors.extend((1 / freq_superior).flatten())
#
#     ymax = max(ax.get_ylim())
#     if var is not None:
#         if important_periods is None:
#             important_periods = {var: period_of_superiors}
#         else:
#             important_periods[var] = period_of_superiors
#
#     # unique_periods = set(period_of_superiors) - set(common_periods) if common_periods else period_of_superiors
#     # common_periods = set(period_of_superiors) & set(common_periods) if common_periods else []
#
#
#     # ax.scatter(, [ymax] * len(period_of_superiors), color='b', alpha=.2, marker='.', s=10, label='Superior to AR1')
#
#     # Scatter for common periods
#     # ax.scatter(outside_periods, [ymax] * len(outside_periods), color='r', marker='.', alpha=.2, s=10, label='elsewhere superior to AR1')
#
#     ax.set_ylim([10, ymax])
#     ax.yaxis.tick_right()
#     ax.set_ylabel('PSD using Welch', labelpad=15, rotation=270)
#     ax.yaxis.set_label_position("right")
#     ax.spines['right'].set_position(('outward', 10))
#     ax.spines['right'].set_visible(True)  # Ensure the right spine is visible
#     ax.spines['left'].set_visible(False)
#     # ax.legend(loc='upper right')  # Add legend to distinguish between the scatters
#
#     return important_periods, ax

def process_group(args):
    grp_d, plot_flag, override_flag, datetime_flag = args
    print(grp_d)

    shifted = prepare_data(grp_d)

    target_ps_list, target_ps_trend_list, target_psd_list = [], [], []
    col_ps_list, col_ps_trend_list, col_psd_list = [], [], []

    for start in range(grp_d['tau']):
        print(f'Starting at {start}')
        subset = shifted.iloc[start:][[grp_d['target_var'], grp_d['col_var'], 'date']][::grp_d['tau']].reset_index(drop=True)

        target_series, target_detrended, target_psd = compute_psd(subset, grp_d['target_var'],
                                                                  f'start: {shifted.loc[start, "date"]}')
        col_series, col_detrended, col_psd = compute_psd(subset, grp_d['col_var'],
                                                         f'start: {shifted.loc[start, "date"]}')

        target_ps_list.append(target_series)
        target_ps_trend_list.append(target_detrended)
        target_psd_list.append(target_psd)

        col_ps_list.append(col_series)
        col_ps_trend_list.append(col_detrended)
        col_psd_list.append(col_psd)

    target_ms = pyleo.MultipleSeries(target_ps_list)
    col_ms = pyleo.MultipleSeries(col_ps_list)

    temp_author = grp_d['temp_author']
    tsi_author = grp_d['tsi_author']

    res_fig_dir = calc_location / f'res_fig_shift0'
    res_fig_dir.mkdir(exist_ok=True, parents=True)

    # # Identify common periods between target and col variables
    # target_super_periods = [1 / freq for psd in target_psd_list for freq in psd.frequency]
    # col_super_periods = [1 / freq for psd in col_psd_list for freq in psd.frequency]
    plot=False
    important_periods = {}
    # Plotting the target variable
    fig_tar, subfigs_tar, axs0_tar = create_fig(target_ms, target_ps_trend_list)
    add_lineplot(axs0_tar[0], temp_author, tsi_author, grp_d['tau'], grp_d['target_var'])
    important_periods, axs0_tar[1] = plot_psd(axs0_tar[1], target_psd_list, var=grp_d['target_var'],
                                              important_periods=important_periods)

    # Plotting the col variable
    fig_col, subfigs_col, axs0_col = create_fig(col_ms, col_ps_trend_list)
    add_lineplot(axs0_col[0], temp_author, tsi_author, grp_d['tau'], grp_d['col_var'])
    important_periods, axs0_col[1] = plot_psd(axs0_col[1], col_psd_list, var=grp_d['col_var'],
                                              important_periods=important_periods)

    # Scatter plot after setting axis limits
    if plot == True:
        yticks = np.array(axs0_tar[1].get_yticks())#get_ylim()[1]
        max_ind = np.argwhere(yticks< axs0_tar[1].get_ylim()[1])[-1]
        ymax = yticks[max_ind]
    if plot ==False:
        ymax=-.5
        axs0_tar2 = axs0_tar[1].twinx()

    for var, color in zip(important_periods.keys(), ['b', 'r']):
        axs0_tar2.scatter(important_periods[var], [ymax] * len(important_periods[var]), color=color, marker='.', alpha=.2, s=100,
                            label=f'{var} superior to AR1')


    if plot == True:
        h, l = axs0_tar[1].get_legend_handles_labels()
        axs0_tar[1].legend(h[-3:], l[-3:], loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)
    if plot == False:
        h, l = axs0_tar2.get_legend_handles_labels()

        axs0_tar2.legend(h[-2:], l[-2:], loc='upper left', bbox_to_anchor=(0.1, 1), frameon=False)
        axs0_tar[1].get_legend().remove()
        for line in axs0_tar[1].lines:
            line.remove()

        axs0_tar2.spines[['left']].set_visible(False)
        axs0_tar2.spines[['right']].set_visible(False)
        axs0_tar2.set_yticks([])
        axs0_tar2.set_ylabel('')
        axs0_tar2.grid(False)

        axs0_tar[1].set_yticks([])
        axs0_tar[1].set_ylabel('')
        axs0_tar[1].grid(False)
        axs0_tar[1].spines[['right']].set_visible(False)

    # axs0_tar[1].set_ylim((-1, 0))
    # print('axs0_tar[1]', axs0_tar[1].get_ylim())



    if plot == True:
        yticks = np.array(axs0_col[1].get_yticks())#get_ylim()[1]
        max_ind = np.argwhere(yticks< axs0_col[1].get_ylim()[1])[-1]
        ymax = yticks[max_ind]
    else:
        axs0_col2 = axs0_col[1].twinx()
        ymax=-.5

    for var, color in zip(important_periods.keys(), ['b', 'r']):
        axs0_col2.scatter(important_periods[var], [ymax] * len(important_periods[var]), color=color, marker='.', alpha=.2, s=100,
                            label=f'{var} superior to AR1')
    if plot == False:
        axs0_col2.set_ylim([-1, 0])


    # axs0_col[1].legend(h[-3:], l[-3:], loc='upper left', bbox_to_anchor=(1.1, 1))
    #
    if plot == True:
        h, l = axs0_col[1].get_legend_handles_labels()
        axs0_col[1].legend(h[-3:], l[-3:], loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)

    if plot == False:
        # axs0_tar[1].lines.clear()
        for line in axs0_col[1].lines:
            line.remove()
        h, l = axs0_col2.get_legend_handles_labels()

        axs0_col2.legend(h[-2:], l[-2:], loc='upper left', bbox_to_anchor=(0.1, 1), frameon=False)
        axs0_col[1].get_legend().remove()

        axs0_col2.spines[['left']].set_visible(False)
        axs0_col2.spines[['right']].set_visible(False)

        axs0_col2.set_yticks([])
        axs0_col2.set_ylabel('')
        axs0_col2.grid(False)
        axs0_col[1].set_yticks([])
        axs0_col[1].set_ylabel('')
        axs0_col[1].grid(False)
        axs0_col[1].spines[['right']].set_visible(False)


    subfigs_tar[1].subplots_adjust(left=0, bottom=None, right=1, top=None, wspace=1, hspace=None)
    fig_tar.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["target_var"]}.png',
                        bbox_inches='tight')

    subfigs_col[1].subplots_adjust(left=0, bottom=None, right=1, top=None, wspace=1, hspace=None)
    fig_col.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["col_var"]}.png',
                bbox_inches='tight')

    important_periods_df = pd.concat([pd.DataFrame({'period':important_periods[key], 'var':[key for _ in important_periods[key]]}) for key in important_periods.keys()])
    important_periods_df['temp_author'] = temp_author
    important_periods_df['tsi_author'] = tsi_author
    important_periods_df['tau'] = grp_d['tau']

    if (res_fig_dir/'periodicities.csv').exists():
        important_periods_df.to_csv(res_fig_dir/'periodicities.csv', mode='a', header=False)
    else:
        important_periods_df.to_csv(res_fig_dir/'periodicities.csv', mode='w')


# def process_group(args):
#     grp_d, plot_flag, override_flag, datetime_flag = args
#     print(grp_d)
#
#     shifted = prepare_data(grp_d)
#
#     target_ps_list, target_ps_trend_list, target_psd_list = [], [], []
#     col_ps_list, col_ps_trend_list, col_psd_list = [], [], []
#
#     for start in range(grp_d['tau']):
#         print(f'Starting at {start}')
#         subset = shifted.iloc[start:][[grp_d['target_var'], grp_d['col_var'], 'date']][::grp_d['tau']].reset_index(
#             drop=True)
#
#         target_series, target_detrended, target_psd = compute_psd(subset, grp_d['target_var'],
#                                                                   f'start: {shifted.loc[start, "date"]}')
#         col_series, col_detrended, col_psd = compute_psd(subset, grp_d['col_var'],
#                                                          f'start: {shifted.loc[start, "date"]}')
#
#         target_ps_list.append(target_series)
#         target_ps_trend_list.append(target_detrended)
#         target_psd_list.append(target_psd)
#
#         col_ps_list.append(col_series)
#         col_ps_trend_list.append(col_detrended)
#         col_psd_list.append(col_psd)
#
#     target_ms = pyleo.MultipleSeries(target_ps_list)
#     col_ms = pyleo.MultipleSeries(col_ps_list)
#
#     temp_author = grp_d['temp_author']
#     tsi_author = grp_d['tsi_author']
#
#     res_fig_dir = calc_location / f'res_fig_shift0'
#     res_fig_dir.mkdir(exist_ok=True, parents=True)
#
#     # Identify common periods between target and col variables
#     target_super_periods = [1 / freq for psd in target_psd_list for freq in psd.frequency]
#     col_super_periods = [1 / freq for psd in col_psd_list for freq in psd.frequency]
#     # common_periods = target_super_periods & col_super_periods
#
#     important_periods = {}
#     # Plotting the target variable
#     fig_tar, subfigs_tar, axs0_tar = create_fig(target_ms)
#     add_lineplot(axs0_tar[0], temp_author, tsi_author, grp_d['tau'], grp_d['target_var'])
#     important_periods, axs0_tar[1] = plot_psd(axs0_tar[1], target_psd_list, var=grp_d['target_var'], important_periods=important_periods)
#
#
#     # Plotting the col variable
#     fig_col, subfigs_col, axs0_col = create_fig(col_ms)
#     add_lineplot(axs0_col[0], temp_author, tsi_author, grp_d['tau'], grp_d['col_var'])
#     important_periods, axs0_col[1]= plot_psd(axs0_col[1], col_psd_list,var=grp_d['col_var'],
#                                              important_periods=important_periods)
#     # print(important_periods)
#     # Scatter for unique periods
#     ymax = 10#max(axs0_tar[1].get_yticks())
#     axs0_tar[1].get_legend().remove()
#     for var, color in zip(important_periods.keys(), ['b', 'r']):
#         print(important_periods[var])
#         axs0_tar[1].scatter(important_periods[var], [ymax for ik in range(len(important_periods[var]))], color=color, marker='.', s=10,
#                    label=f'{var} superior to AR1')
#
#     ymax = 10#max(axs0_col[1].get_yticks())
#     axs0_col[1].get_legend().remove()
#     for var, color in zip(important_periods.keys(), ['b', 'r']):
#         axs0_col[1].scatter(important_periods[var], [ymax] * len(important_periods[var]),
#                             color=color, marker='.', s=10,zorder=100,
#                             label=f'{var} superior to AR1')
#     # axs0_col2.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
#
#     fig_tar.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["target_var"]}.png',
#                         bbox_inches='tight')
#
#     fig_col.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["col_var"]}.png',
#                 bbox_inches='tight')

def create_fig(multiple_series, trend_ps_list):
    """Create a figure and subfigures for plotting."""

    # Determine the optimal figure height
    num_series = len(multiple_series)

    # Dynamically calculate height ratios and hspace
    if num_series == 0:
        height_ratio_top = 1
    else:
        height_ratio_top = max(1, num_series / 7)  # Scale with respect to 7 being optimal

    height_ratio_bottom = 3  # Fixed height for the bottom subplot
    total_height_ratio = height_ratio_top + height_ratio_bottom

    # Normalize ratios to ensure they add up to the total height (could also use just `[height_ratio_top, height_ratio_bottom]`)
    height_ratios = [height_ratio_top / total_height_ratio, height_ratio_bottom / total_height_ratio]

    # Adjust hspace dynamically
    hspace = -0.2 * (7 / num_series) if num_series != 0 else 0.2

    # Create the figure and subfigures
    fig = plt.figure(figsize=(6, num_series + 2))
    subfigs = fig.subfigures(2, 1, height_ratios=height_ratios, hspace=hspace)


    # fig = plt.figure(figsize=(6, len(multiple_series) + 2))
    # subfigs = fig.subfigures(2, 1, height_ratios=[len(multiple_series), 3], hspace=-.2)
    subfigs[0], ax_d = multiple_series.stackplot(plot_kwargs={'marker': 'x', 'linewidth': 1, 'ms': 2}, fig=subfigs[0])
    for ik in range(len(multiple_series)):
        ax = ax_d[ik]
        trend_ps = trend_ps_list[ik]
        # for ik, target_trend_ps in enumerate(target_ps_trend_list):
        trend_ps.plot(ax=ax_d[ik], zorder=-10, color='k',alpha=.5, linewidth=1, xlabel='',legend=False)
        # ax_d[ik].set_ylim([-1.1, 1.1])
        # ax.set_ylim([-1.1, 1.1])

    axs0 = subfigs[1].subplots(1, 2, gridspec_kw=dict(wspace=.1))
    return fig, subfigs, axs0


def add_lineplot(ax, temp_author, tsi_author, tau, var):
    """Add a line plot to the axis."""
    self_corr_df = pd.read_csv(calc_location / 'self_corr' / 'self_corr.csv')
    self_corr_df = self_corr_df[(self_corr_df['temp_author'] == temp_author) &
                                (self_corr_df['tsi_author'] == tsi_author) &
                                (self_corr_df['tau'] == tau) &
                                (self_corr_df['var'] == var)]
    sns.lineplot(data=self_corr_df, x='E', y='avg_self_corr', hue='Tp', ax=ax)


def plot_periodicities(res_fig_dir):
    if (res_fig_dir / 'periodicities.csv').exists():
        periodicity_dir = res_fig_dir/'periodicities'
        periodicity_dir.mkdir(exist_ok=True, parents=True)
        important_periods_df = pd.read_csv(res_fig_dir / 'periodicities.csv', index_col=0)
        print(important_periods_df.head())
        tsi_df = important_periods_df[important_periods_df['var'] == 'TSI'].copy()
        tsi_df = tsi_df.drop(columns=['temp_author'])
        tsi_df = tsi_df.rename(columns={'tsi_author': 'author'})

        temp_df = important_periods_df[important_periods_df['var'] == 'temp'].copy()
        temp_df = temp_df.drop(columns=['tsi_author'])
        temp_df = temp_df.rename(columns={'temp_author': 'author'})
        authors_grp_df = pd.concat([tsi_df, temp_df])
        # for authors, authors_grp_df in important_periods_df.groupby(['temp_author', 'tsi_author']):
        # temp_author, tsi_author = authors
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        authors_grp_df['small_large'] = authors_grp_df['period'].apply(lambda x: 'small' if x < 1000 else 'large')
        authors_grp_df.sort_values(by=['small_large', 'var'], inplace=True, ascending=False)
        g = sns.FacetGrid(authors_grp_df, col="small_large", row="tau", hue="author", aspect=3,height=3,sharex=False,
                          sharey=False,
                          palette='tab10', gridspec_kws={'width_ratios': [6, .5]})
        g.map_dataframe(sns.kdeplot, x="period", common_norm=False, clip_on=False,
                        lw=.15, bw_adjust=.1, fill=True)
        g.add_legend()#title=f'{temp_author}, {tsi_author}')
        for ik in range(len(g.axes)):
            g.axes[ik,0].set_xlim([50, 600])
            g.axes[ik,0].set_ylim([0, .15])

            g.axes[ik, 1].set_xlim([1000, 8000])
        # g.set_title(f'{temp_author} vs {tsi_author} Periodicities')
        plt.savefig(periodicity_dir / f'all_periodicities.png', bbox_inches='tight')

        # g = sns.FacetGrid(authors_grp_df, row="tau", col='small_large', hue="author",#xlim=[(0, 1000), (1000, 8000)],
        #                   aspect=5, height=1, palette='tab10',  sharex=True, gridspec_kws={'width_ratios': [3, 2]})
        #
        # # Draw the densities in a few steps
        # g.map(sns.kdeplot, "period",common_norm=False,#cut=-2,
        #       bw_adjust=.5, clip_on=False,
        #       fill=True, linewidth=.1)
        # g.map(sns.kdeplot, "period", common_norm=False, clip_on=False, color="w", lw=.15, bw_adjust=.5)
        #
        # # passing color=None to refline() uses the hue mapping
        # g.refline(y=0, linewidth=.5, linestyle="-", clip_on=False)
        # for ax in g.axes.flatten():
        #     # xlims = ax.get_xlim()
        #     # print(xlims)
        #     # min_x = xlims[0]
        #     # max_x = xlims[1]
        #     # if xlims[1] >2000:
        #     #     min_x = 1000
        #     #     max_x = 8000
        #     # else:
        #     #     min_x = 0
        #     #     max_x = 1000
        #     # print([min_x, max_x])
        #     # ax.set_xlim([min_x, max_x])
        #     ax.set_facecolor('none')
        #
        # # Define and use a simple function to label the plot in axes coordinates
        # # Function to label the plot with tau values
        # def label_tau(x, color, label, **kwargs):
        #     ax = plt.gca()
        #     label_split = ax.get_title().split('|')
        #     if 'small' in label_split[-1]:
        #         tau_value = label_split[0].strip()  # Extract tau value from title
        #         ax.text(0, .2, f'{tau_value}', fontweight="bold",
        #                 ha="left", va="center", transform=ax.transAxes)
        #     else:
        #         ax.text(0, .2, '', fontweight="bold",
        #                 ha="left", va="center", transform=ax.transAxes)
        #         # ax.set_xlim([-10, 1000])
        #     # else:
        #     #     ax.set_xlim([1000, 8000])
        #
        # # Map the labeling function to each subplot
        # g.map(label_tau, "tau")
        #
        # # Set the subplots to overlap
        # g.figure.subplots_adjust(hspace=-.5)
        #
        # # Remove axes details that don't play well with overlap
        # g.set_titles("")
        # g.set(yticks=[], ylabel="")
        # g.despine(bottom=True, left=True)
        # plt.savefig(periodicity_dir / f'all_periodicities2.png', bbox_inches='tight')

            # g= sns.displot(data=authors_grp_df, x='period',facet_kws=dict(sharex='col'),
            #                row='var',col='small_large', hue='tau', palette='tab20')
            # for ip in [0,1]:
            #     g.axes[ip,0].set_xlim([2000,8100])
            #     g.axes[ip,1].set_xlim([10, 2000])
            # g.add_legend(title=f'{temp_author}, {tsi_author}')
            # g.set_title(f'{temp_author} vs {tsi_author} Periodicities')
            # plt.savefig(periodicity_dir / f'{temp_author}_{tsi_author}_periodicities.png', bbox_inches='tight')


if __name__ == '__main__':
    calc_grps_path = calc_carc / 'calc_grps.csv'
    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[['temp_author', 'tsi_author', 'tau', 'target_var', 'col_var', 'Tp']].drop_duplicates()
    calc_grps = calc_grps_df.to_dict(orient='records')
    res_fig_dir = calc_location / f'res_fig_shift0'

    # plot_flag = False
    # override = True
    # datetime_flag = None
    #
    # args = [(grp_d, plot_flag, override, datetime_flag) for grp_d in calc_grps]
    # num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([3, len(args)])))
    #
    # with Pool(num_cpus) as pool:
    #     results = pool.map(process_group, args)

    plot_periodicities(res_fig_dir)




# from multiprocessing import Pool
# import re
# import os
# from collections import Counter
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import sys
# import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pyEDM as pe
# import pyleoclim as pyleo
#
# import warnings
# warnings.simplefilter("ignore", category=FutureWarning)
#
# # Define directories for notebooks and calculations
# notebooks_dir = Path(os.getcwd()) / 'notebooks'
# calc_carc = notebooks_dir / 'calc_carc'
#
# calc_metric_dir_name = 'calc_metrics_scaled'
# if Path('/Users/jlanders').exists() == True:
#     calc_location = notebooks_dir / 'calc_local_tmp'
# else:
#     calc_location = notebooks_dir / 'calc_carc'
#
# pd.option_context('mode.use_inf_as_na', True)
#
# proj_d = {'erb':{'tau':[10, 5], 'temp_var':'erb2022_da_temp', 'E':[6, 9], 'surr_file_name':'pe_hol_tsi_temp_surr_vieira_erb.csv'},
#           'essel':{'tau':[5, 10], 'temp_var':'essel2023_opt_temp', 'E':[14], 'surr_file_name':'pe_hol_tsi_temp_surr_wu_essel.csv'},
#           'vieira':{'tau':[5, 10], 'tsi_var':'vieira_tsi', 'surr_file_name':'pe_hol_tsi_temp_surr_vieira_erb.csv'},
#           'wu':{'tau': [5, 10], 'tsi_var':'wu_tsi', 'surr_file_name':'pe_hol_tsi_temp_surr_wu_essel.csv'}
#           }
#
# def process_group(args):
#     grp_d, plot_flag, override_flag, datetime_flag = args
#     print(grp_d)
#
#     temp_var = proj_d[grp_d['temp_author']]['temp_var']#'erb2022_da_temp'#'essel2023_opt_temp'
#     tsi_var = proj_d[grp_d['tsi_author']]['tsi_var']#'vieira_tsi'#'wu_tsi'
#
#     # load data
#     data = pd.read_csv(notebooks_dir.parent / 'data' / 'hol_ct_data_short.csv', index_col=0)
#     data = data[['time', temp_var, tsi_var]].rename(columns={'time': 'date', temp_var: 'temp', tsi_var: 'TSI'})
#
#     # shift data per usual
#     shifted = data.copy()
#     shift=0
#     shifted[grp_d['target_var']] = shifted[grp_d['target_var']].shift(shift)
#     shifted = shifted.dropna(subset=[grp_d['target_var'], grp_d['col_var']])
#     shifted = shifted.reset_index(drop=True)
#
#     # Plotting the time series at different resolutions starting at each point from 0 to tau-1
#     target_ps_list= []
#     target_ps_trend_list = []
#     target_psd_list = []
#     col_ps_trend_list = []
#     col_ps_list = []
#     col_psd_list = []
#     for start in range(0,grp_d['tau']):
#         print(f'Starting at {start}')
#         subset = shifted.iloc[start:]
#         subset = subset[[grp_d['target_var'], grp_d['col_var'], 'date']][::grp_d['tau']].reset_index(drop=True) # Subset the data starting at 'start' with step 'tau'
#         # subset = subset[::grp_d['tau']].reset_index(drop=True)  # Subset the data starting at 'start' with step 'tau'
#
#         ac_val = pyleo.utils.tsmodel.ar1_fit(subset[grp_d['target_var']])
#         target_ps = pyleo.Series(time=subset['date'], value=subset[grp_d['target_var']], value_name=f'start: {shifted.loc[start,"date"]}' , label=f'{ac_val:.3f}', time_name='years')
#         # ac_val = pyleo.utils.tsmodel.ar1_fit(subset[grp_d['target_var']])
#         _,detrending = pyleo.utils.tsutils.detrend(target_ps.value, method='emd')
#         # print('detrending1', detrending[0])
#         print(len(detrending), len(target_ps.time))
#         detrend_ps = pyleo.Series(time=target_ps.time, value=detrending,
#                      value_name=f'start: {shifted.loc[start, "date"]}-trend', label=f'{grp_d["target_var"]}',
#                      time_name='years')
#         psd_welch = target_ps.spectral(method='welch')
#         target_psd_list.append(psd_welch)
#         target_ps_trend_list.append(detrend_ps)
#         target_ps_list.append(target_ps)
#
#         ac_val = pyleo.utils.tsmodel.ar1_fit(subset[grp_d['col_var']])
#         col_ps = pyleo.Series(time=subset['date'], value=subset[grp_d['col_var']], value_name=f'start: {shifted.loc[start,"date"]}', label=f'{ac_val:.3f}', time_name='years')
#         # plt.plot(subset, label=f'Start at {start}')
#         _,detrending = pyleo.utils.tsutils.detrend(col_ps.value, method='emd')
#         detrend_ps = pyleo.Series(time=col_ps.time, value=detrending,
#                      value_name=f'start: {shifted.loc[start, "date"]}-trend', label=f'{grp_d["col_var"]}',
#                      time_name='years')
#         psd_welch = col_ps.spectral(method='welch')
#         col_psd_list.append(psd_welch)
#         col_ps_trend_list.append(detrend_ps)
#         col_ps_list.append(col_ps)
#
#     target_ms = pyleo.MultipleSeries(target_ps_list)
#     col_ms = pyleo.MultipleSeries(col_ps_list)
#
#     temp_author = grp_d['temp_author']
#     tsi_author = grp_d['tsi_author']
#
#     res_fig_dir = calc_location / f'res_fig_shift{shift}'
#     res_fig_dir.mkdir(exist_ok=True, parents=True)
#
#     # fig = plt.figure(figsize=(8, target_ms))
#     fig = plt.figure(figsize=(6, len(target_ms)+2))
#     subfigs = fig.subfigures(2, 1, height_ratios=[len(target_ms),3], hspace=-.2)
#     # fig.suptitle(f'{temp_author} vs {tsi_author} Cross Correlation', fontsize=16)
#     subfigs[0], ax_d = target_ms.stackplot(plot_kwargs={'marker':'x', 'linewidth':1, 'ms':2, }, fig=subfigs[0])
#     for ik, target_trend_ps in enumerate(target_ps_trend_list):
#         target_trend_ps.plot(ax=ax_d[ik], zorder=-10, color='k',alpha=.5, linewidth=1, xlabel='',legend=False)
#         ax_d[ik].set_ylim([-1.1, 1.1])
#
#     axs0 = subfigs[1].subplots(1, 2, gridspec_kw=dict(wspace=.1))
#     line_plot = axs0[0]
#     self_corr_df = pd.read_csv(res_fig_dir.parent/'self_corr'/'self_corr.csv')
#     self_corr_df = self_corr_df[(self_corr_df['temp_author']==temp_author) &
#                                 (self_corr_df['tsi_author']==tsi_author) &
#                                 (self_corr_df['tau']==grp_d['tau'])&
#                                 (self_corr_df['var']==grp_d['target_var'])]
#     sns.lineplot(data=self_corr_df, x='E', y='avg_self_corr', hue='Tp', ax=line_plot)
#
#     psd_plot = axs0[1]
#     period_of_superiors=[]
#     for psd_welch in target_psd_list:
#         psd_welch_signif = psd_welch.signif_test(number=200, method='ar1sim')  # in practice, need more AR(1) simulations
#         psd_welch_signif.plot(title='PSD using Welch method', ax=psd_plot, linewidth=2, legend=False)
#         freq = psd_welch_signif.frequency
#         indexes_of_superior = np.argwhere(psd_welch_signif.amplitude > psd_welch_signif.signif_qs.psd_list[0].amplitude)
#         freq_superior = freq[indexes_of_superior]
#         period_of_superior = 1 / freq_superior
#         period_of_superiors.append(period_of_superior)
#
#     period_of_superior = np.concatenate(period_of_superiors).flatten()
#     ylim = psd_plot.get_ylim()
#     psd_plot.set_ylim([10, ylim[1]])
#
#     ymax = max(psd_plot.get_ylim())
#     psd_plot.scatter(period_of_superior, [ymax] * len(period_of_superior), color='k', marker='.', s=10,
#                label='Superior to AR1')
#
#     target_super_periods = Counter(list(period_of_superior))
#
#
#     psd_plot.yaxis.tick_right()
#     psd_plot.set_ylabel('PSD using Welch', labelpad=15, rotation=270)
#     psd_plot.yaxis.set_label_position("right")
#     psd_plot.spines['right'].set_position(('outward', 10))
#     psd_plot.spines['left'].set_visible(False)
#
#
#     psd_plot.set_title('')
#     # remove psd_plot legend
#     leg = psd_plot.get_legend()
#     if leg:
#         leg.remove()
#
#     plt.subplots_adjust(left=0, bottom=None, right=1, top=None, wspace=1, hspace=None)
#     fig.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["target_var"]}.png', bbox_inches='tight')
#
#
#     ### Col var
#     fig = plt.figure(figsize=(6, len(target_ms) + 2))
#     subfigs = fig.subfigures(2, 1, height_ratios=[len(target_ms), 3], hspace=-.2)
#     # fig.suptitle(f'{temp_author} vs {tsi_author} Cross Correlation', fontsize=16)
#     subfigs[0], ax_d = col_ms.stackplot(plot_kwargs={'marker': 'x', 'linewidth': 1, 'ms': 2, }, fig=subfigs[0])
#     for ik, col_trend_ps in enumerate(col_ps_trend_list):
#         col_trend_ps.plot(ax=ax_d[ik], zorder=-10, color='k',alpha=.5, linewidth=1, xlabel='',legend=False)
#         # ax_d[ik].set_ylim([-.3, .3])
#
#     axs0 = subfigs[1].subplots(1, 2, gridspec_kw=dict(wspace=.1))
#     line_plot = axs0[0]
#     self_corr_df = pd.read_csv(res_fig_dir.parent / 'self_corr' / 'self_corr.csv')
#     self_corr_df = self_corr_df[(self_corr_df['temp_author'] == temp_author) &
#                                 (self_corr_df['tsi_author'] == tsi_author) &
#                                 (self_corr_df['tau'] == grp_d['tau'])&
#                                 (self_corr_df['var']==grp_d['col_var'])]
#     sns.lineplot(data=self_corr_df, x='E', y='avg_self_corr', hue='Tp', ax=line_plot)
#
#     period_of_superiors=[]
#     psd_plot = axs0[1]
#     for psd_welch in col_psd_list:
#         psd_welch_signif = psd_welch.signif_test(number=200, method='ar1sim')  # in practice, need more AR(1) simulations
#         psd_welch_signif.plot(title='PSD using Welch method', ax=psd_plot, legend=False)
#         freq = psd_welch_signif.frequency
#         indexes_of_superior = np.argwhere(psd_welch_signif.amplitude > psd_welch_signif.signif_qs.psd_list[0].amplitude)
#         freq_superior = freq[indexes_of_superior]
#         period_of_superior = 1 / freq_superior
#         period_of_superiors.append(period_of_superior)
#
#     period_of_superior = np.concatenate(period_of_superiors).flatten()
#     print(period_of_superior)
#
#     ylim = psd_plot.get_ylim()
#     psd_plot.set_ylim([10, ylim[1]])
#
#     ymax = max(psd_plot.get_ylim())
#     psd_plot.scatter(period_of_superior, [ymax] * len(period_of_superior), color='k', marker='.', s=10,
#                      label='Superior to AR1')
#     col_super_periods = Counter(list(period_of_superior))
#
#
#     psd_plot.set_title('')
#
#     psd_plot.yaxis.tick_right()
#     psd_plot.set_ylabel('PSD using Welch', labelpad=15, rotation=270)
#     psd_plot.yaxis.set_label_position("right")
#     psd_plot.spines['right'].set_position(('outward', 10))
#     psd_plot.spines['left'].set_visible(False)
#
#     # remove psd_plot legend
#     leg = psd_plot.get_legend()
#     if leg:
#         leg.remove()
#     # psd_plot.get_legend.remove()
#     # subfigs[1].subplots_adjust(wspace=1)
#     plt.subplots_adjust(left=0, bottom=None, right=1, top=None, wspace=1, hspace=None)
#
#     fig.savefig(res_fig_dir / f'{temp_author}_{tsi_author}_tau{grp_d["tau"]}_{grp_d["col_var"]}.png', bbox_inches='tight')
#
#
# if __name__ == '__main__':
#
#     calc_grps_path = calc_carc / 'calc_grps.csv'
#     calc_grps_df = pd.read_csv(calc_grps_path)
#     calc_grps_df = calc_grps_df[['temp_author', 'tsi_author', 'tau', 'target_var', 'col_var', 'Tp']].drop_duplicates()
#     calc_grps = calc_grps_df.to_dict(orient='records')
#
#     plot_flag = False
#     override = False
#     datetime_flag = None
#
#     args = [(grp_d, plot_flag, override, datetime_flag) for grp_d in calc_grps][:1]
#
#     num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([3, len(args)])))
#
#     # Use multiprocessing to parallelize the process
#     with Pool(num_cpus) as pool:
#         results = pool.map(process_group, args)
#
#     # # Collect results from parallel processing
#     # df_pathed_dfs = []
#     # for result_lst in results:
#     #     df_pathed_dfs.extend(result_lst)
#     #
#     # calc_runs_df = pd.DataFrame(df_pathed_dfs)
#     # calc_runs_df = calc_runs_df.reset_index(drop=True)
#
#     # cross_corr_dir = calc_location / 'cross_corr'
#     # cross_corr_dir.mkdir(exist_ok=True, parents=True)
#
#     # calc_runs_df.to_csv(cross_corr_dir / 'cross_corr.csv')