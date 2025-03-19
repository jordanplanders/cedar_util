import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pyleoclim as pyleo
import time
from utils.arg_parser import get_parser
from utils.config_parser import load_config

def embedding_explanation(arg_tuple):
    params ={
        'font.size': 24,  # Base font size
        'axes.titlesize': 24,  # Title font size
        'axes.labelsize': 24,  # Axis label font size
        'xtick.labelsize': 24,  # X tick label font size
        'ytick.labelsize': 24,  # Y tick label font size
        'legend.fontsize': 24,  # Legend font size
        'figure.titlesize': 24,  # Figure title font size
        'figure.dpi': 300,  # DPI for publication-quality figures
        'savefig.dpi': 300  # DPI when saving figures
    }
    plt.rcParams.update(params)


    config, data_dir, fig_dir, flags, var, lr = arg_tuple
    pal = config.pal.to_dict()

    data_csv = config.raw_data.data_csv
    time_var = config.raw_data.time_var

    # Read the data
    data = pd.read_csv(data_dir / f'{data_csv}.csv', index_col=0)

    # Select one time series variable
    data_target_vars_d = {config.get_dynamic_attr("{var}.data_var", var_id): var_id for var_id in config.target.ids}
    for var_id in config.col.ids:
        data_target_vars_d[config.get_dynamic_attr("{var}.data_var", var_id)]= var_id

    target_df = data.rename(columns=data_target_vars_d)

    target_label = config.get_dynamic_attr("{var}.var_label", var)
    target_unit = config.get_dynamic_attr("{var}.unit", var)

    # Define the delay and segment length
    n = 1   # Units to shift between segments
    E = 7   # Number of points in each scatterplot
    tau = 8 # Spacing between points
    length = (E-1) * tau  # Length of each segment

    # Extract the time series data
    target_df = target_df[[time_var, var]].dropna()
    time_series = target_df.rename(columns={var: 'value'})

    # Create segments
    segments = []
    start_indices = [0,1,23]
    min_time = time_series[time_var].min()-20
    max_time = time_series.iloc[4:4+length +1][time_var].max()
    for start_index in start_indices:
        # start_index = k
        end_index = start_index + length +1
        segment_df = time_series.iloc[start_index:end_index]
        print(segment_df[time_var].min(), segment_df[time_var].max())
        segments.append(segment_df)
        # start_indices.append(start_index)

    print(start_indices)
    # Adjust the figure size
    figheight = 6 * 1.5  # Adjust as needed
    figwidth = 8

    # Set up the figure with 5 subplots
    fig = plt.figure(figsize=(figwidth, figheight))
    gs = GridSpec(6, 1, figure=fig, hspace=.2)
    total_length = 1 + length +5#end_index + 2
    time_series = time_series.iloc[:total_length].copy()

    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axs.append(fig.add_subplot(gs[5, 0]))

    # Plot the full time series in the top panel
    ax0 = axs[0]
    sns.lineplot(x=time_var, y='value', data=time_series, ax=ax0, color='k', alpha=.2)#pal[target_var])
    ax0.set_ylabel(f'{target_label} [{target_unit}]')
    ax0.set_title('')#(r'$X$')
    ax0.spines[['right', 'top']].set_visible(False)
    ylims = ax0.get_ylim()
    amplitude = max(np.abs(ylims[0]), np.abs(ylims[1]))*1.1
    print('amplitude', amplitude)
    ax0.set_ylim(-amplitude, amplitude)
    yticks = ax0.get_yticks()
    ax0.set_xlim((min_time, max_time))
    _xticks = ax0.get_xticks()
    ax0.spines[['right', 'top', 'left']].set_visible(False)

    ax0_xilms = ax0.get_xlim()
    _xticks = np.linspace(ax0_xilms[0], ax0_xilms[1], 5)
    ax0.set_xticks(ticks=_xticks)
    ax0.grid(False)
    ax0.set_yticks([])
    ax0.set_ylabel('')
    if lr == 'l':
        ax0.text(-0.13, -.05, time_var, transform=ax0.transAxes,fontsize=params['font.size'],ha='left',
                 va='bottom')
    elif lr == 'r':
        ax0.text(1.13, -.05, time_var, transform=ax0.transAxes, fontsize=params['font.size'],ha='right',
                 va='bottom')
    # ax0.set_ylabel('\n'+time_var, loc='bottom', rotation=0, labelpad=33)
    ax0.set_xlabel('')


    #remove xticks
    ax0.set_xticks([])
    # ax0.spines['left'].set_position(('data', yticks[1] - .02*np.abs(yticks[1])))  # Move the left spine to the middle of the plot


    # ax0.spines[['bottom']].set_bounds(time_series[time_var].min(), time_series[time_var].max())

    # Define colors for each set of scatter points
    colors = sns.color_palette("husl", n_colors=4)

    # Plot the segments and overlay scatter points
    min_x = time_series[time_var].min()
    max_x = time_series[time_var].min()
    for i, (segment_df, start_idx) in enumerate(zip(segments, start_indices)):
        ax = axs[i+1]
        ax.plot(segment_df[time_var], segment_df['value'], color='k', alpha=.1, linestyle='--', linewidth=2, zorder=0)
        ax.set_facecolor('none')
        ax.set_ylabel(f'Segment {i+1}')
        ax_tmp_xlim = ax.get_xlim()
        if i<2:
            if ax_tmp_xlim[1] > max_x:
                max_x = ax_tmp_xlim[1]
            if ax_tmp_xlim[0] < min_x:
                min_x = ax_tmp_xlim[0]


        # Overlay scatter points
        # Calculate indices for the scatter points within the segment
        scatter_indices = np.arange(0, E*tau+1, tau)[:E]
        # Ensure indices are within segment bounds
        scatter_indices = scatter_indices[scatter_indices < len(segment_df)]
        # Get times and values for scatter points
        scatter_times = segment_df[time_var].iloc[scatter_indices].values
        if max(scatter_times) > max_time:
            outer_bound = max_time
        else:
            outer_bound = max(scatter_times)
        scatter_time_idx = np.argwhere(scatter_times <= ax0_xilms[1]+50)
        scatter_times = scatter_times[scatter_time_idx]
        scatter_values = segment_df['value'].iloc[scatter_indices].values
        scatter_values = scatter_values[scatter_time_idx]
        # Plot scatter points
        ax.scatter(scatter_times, scatter_values, color=colors[i], s=50, zorder=5, label=f'Set {i+1}')
        ax0.scatter(scatter_times, scatter_values, color=colors[i], s=10,alpha=1, zorder=5)

        # Optionally connect the points
        ax.plot(scatter_times, scatter_values, color=colors[i], linestyle='-', zorder=5)

        ax.spines[['bottom']].set_bounds(min(scatter_times), outer_bound)
        _ax_xticks = [tick for tick in _xticks if tick >= min(scatter_times) and tick <= min(max(scatter_times), outer_bound)]
        if scatter_times[0]-_ax_xticks[0] <30:
            _ax_xticks = _ax_xticks[1:]
        if _ax_xticks[-1]-scatter_times[-1] <30:
            _ax_xticks = _ax_xticks[:-1]

        ax_xticks = [scatter_times[0]]
        ax_xticks.append(scatter_times[-1])
        ax_xticks = np.concatenate(ax_xticks)
        # ax.set_xticks(ticks=ax_xticks)
        ax.set_yticks([])
        ax.set_ylabel('')

        ax.set_ylim(-amplitude, amplitude)
        ax.set_xlim(ax0_xilms)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.spines[['bottom']].set_bounds(min(scatter_times), outer_bound)
        print('outer_bound', outer_bound)
        ax.grid(False)
        print('scatter_times[0]', scatter_times[0])
        # ax.annotate('', xy=(scatter_times[0][0], scatter_values[0][0]), xytext=(scatter_times[0], scatter_values[0]),
        #              xycoords=ax0.transData, textcoords=ax.transData,
        #              arrowprops=dict(arrowstyle='-', color=colors[i], lw=2),
        #              annotation_clip=False)
        ax.annotate('', xy=(scatter_times[0][0], scatter_values[0][0]), xytext=(scatter_times[0][0], scatter_values[0][0]),
                    xycoords=ax0.transData, textcoords=ax.transData,
                    arrowprops=dict(arrowstyle='-', color=colors[i], lw=1, linestyle='--'),
                    annotation_clip=False)
        if start_idx==1:
        #     ax.annotate(
        #         'Time delay embedding',
        #         xy=(scatter_times[-1][0], min(ax.get_ylim())),  # Arrow destination on axis
        #         xytext=(.7, -0.5),  # Label positioned near bottom right in axis coords
        #         xycoords='data',  # Arrowhead in data coordinates
        #         textcoords='axes fraction',  # Label relative to the entire plot
        #         ha='right',  # Align text to the right
        #         va='bottom',  # Vertically center-align text
        #         arrowprops=dict(
        #             arrowstyle='->',  # Ensure arrowhead points correctly
        #             color=colors[i],
        #             lw=1,
        #             linestyle='--'
        #         ),
        #         annotation_clip=False  # Ensure drawing outside plot is allowed
        #     )

            from matplotlib import colors as mc

            # Rest of your code

            fc = mc.to_rgba('white')
            fc = fc[:-1] + (0.5,)
            bbox = dict(boxstyle="round", fc='w', ec="none", alpha=0.8)

            # if i==0:
                # ax_spectral.annotate(
                #     'Cycle x',
                #     xy=(506 + 1, .5),
                #     xytext=(750 + 1, 1.15),
                #     ha='right',  # Horizontal alignment
                #     va='bottom',  # Vertical alignment
                #     arrowprops=dict(
                #         arrowstyle="->",
                #         connectionstyle="angle,angleA=0,angleB=120"
                #     )
                # )

            if lr == 'l':
                yloc =min(ax.get_ylim())
                ax.annotate(
                    r'$t_{1} + (E - 1) * \tau\Delta t$',
                    xy=(scatter_times[-1][0], yloc),
                    xytext=(.94,-1.1),#scatter_times[-1][0], min(ax.get_ylim())-.1),
                    xycoords='data',  # Arrowhead in data coordinates
                    textcoords='axes fraction',  # Label relative to the entire plot
                    ha='right',  # Horizontal alignment
                    va='top',  # Vertical alignment
                    arrowprops=dict(
                        arrowstyle="-",
                        connectionstyle="angle,angleA=0,angleB=90"
                    ),
                    bbox=bbox
                )
        if start_idx>0:
            ax.set_xticks(ticks=[ax_xticks[0]], labels=[r'$t_{{{}}}$'.format(start_idx)])

        else:
            ax.set_xticks(ticks=ax_xticks, labels = [r'$t_{{{}}}$'.format(start_idx), ''])
        if lr == 'r':
            if start_idx<10:
                ax.set_xticks(ticks=ax_xticks, labels=[r'$t_{{{}}}$'.format(start_idx), ''])
        for label in ax.get_xticklabels():
            label.set_ha('right')

    # ax0.set_ylabel('')
    plt.tight_layout()

    # Save the figure
    raw_fig_id = f"{var}_time_delay_embedding"
    fig_target_path = fig_dir / f'{raw_fig_id}.png'
    plt.savefig(fig_target_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

def find_streaks(indexes, delta=2):
    # if not indexes:
    #     return []

    streaks = []
    current_streak = [indexes[0]]

    for i in range(1, len(indexes)):
        if indexes[i] - indexes[i - 1] < delta:
            current_streak.append(indexes[i])
        else:
            streaks.append(current_streak)
            current_streak = [indexes[i]]

    streaks.append(current_streak)
    streaks = [np.array(streak).flatten() for streak in streaks]
    return streaks


from matplotlib.ticker import FuncFormatter

def plot_raw(arg_tuple):
    params ={
        'font.size': 14,  # Base font size
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 15,  # Axis label font size
        'xtick.labelsize': 13,  # X tick label font size
        'ytick.labelsize': 13,  # Y tick label font size
        'legend.fontsize': 14,  # Legend font size
        'figure.titlesize': 18,  # Figure title font size
        'figure.dpi': 300,  # DPI for publication-quality figures
        'savefig.dpi': 300  # DPI when saving figures
    }
    plt.rcParams.update(params)

    def custom_formatter(x, pos):
        fixed = ('%f' % x).rstrip('0').rstrip('.')
        fixed = fixed.replace('0.', '.')
        return fixed
        # return ('%f' % x).rstrip('0').rstrip('.')

    # Apply the custom formatter to the x and y axis
    # ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    config, data_dir, fig_dir, plot_list = arg_tuple
    pal = config.pal.to_dict()

    data_csv = config.raw_data.data_csv
    time_var = config.raw_data.time_var
    legend = True
    legend_flag = '_legend'
    data = pd.read_csv(data_dir / f'{data_csv}.csv', index_col=0)

    data_col_vars_d = {config.get_dynamic_attr("{var}.data_var", var_id):var_id for var_id in config.col.ids}
    stds = data[data_col_vars_d.keys()].std()
    col_ys = 1
    if len(stds)==2:
        log_ratio = int(np.log10(stds[0] / stds[1]))
        if log_ratio != 0:
            col_ys=2

    col_df = data.rename(columns=data_col_vars_d)
    col_df = pd.melt(col_df, id_vars=[time_var], value_vars=config.col.ids, var_name='source', value_name='value')
    col_df['var'] = config.col.var

    data_target_vars_d = {config.get_dynamic_attr("{var}.data_var", var_id):var_id for var_id in config.target.ids}
    stds = data[data_target_vars_d.keys()].std()
    target_ys = 1
    if len(stds) == 2:
        log_ratio = int(np.log10(stds[0] / stds[1]))
        if log_ratio != 0:
            target_ys = 2

    target_df = data.rename(columns=data_target_vars_d)
    target_df = pd.melt(target_df, id_vars=[time_var], value_vars=config.target.ids, var_name='source', value_name='value')
    target_df['var'] = config.target.var

    raw_fig_id = config.col.ids
    raw_fig_id.extend(config.target.ids)
    raw_fig_id = '_'.join(raw_fig_id)

    dt = int(target_df.time.diff().mean())
    plot_label = '_'.join(plot_list)
    fig_name_leg = f'{raw_fig_id}_dt{dt}_{plot_label}{legend_flag}.pdf'
    fig_name_noleg = f'{raw_fig_id}_dt{dt}_{plot_label}.pdf'

    # fig_target_path = fig_dir / fig_name

    figwidth = len(plot_list)
    width_ratios = []
    if 'raw_ts' in plot_list:
        figwidth += 10
        width_ratios.append(5)

    if 'raw_periodogram' in plot_list:
        figwidth += 7
        width_ratios.append(4)

    vars = [config.col.var, config.target.var]
    figheight = int(.7*(len(vars) * 4+2))

    if len(width_ratios) == 1:
        width_ratios = [1]
        hspace = -0.05
        wspace=None
    else:
        hspace = 0.13
        wspace = .25

    fig = plt.figure(figsize=(figwidth, figheight))
    right=0.95
    if legend == True:
        right=0.85

    gs = GridSpec(len(vars)+2, len(plot_list), figure=fig, width_ratios=width_ratios,
                  height_ratios=[4,.5,.15,4],
                  hspace=hspace,right=right, wspace=wspace)

    # subgrid = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[j + 2, i], height_ratios=[4, 2])
    # Top subplot for the time series segment
    linewidth = 1
    alpha=0.7

    time_unit = config.get_dynamic_attr("raw_data.{var}", 'time_unit')

    legend_ax=None
    handles, labels = [], []
    axs = []
    xticks_ts = None
    psds_d = {}
    print(xticks_ts)
    x_text_loc = -.07
    y_text_loc = .99
    text_size = 16
    loc_d = {'[TL]':'(a)', '[BL]':'(b)', '[TR]':'(c)', '[BR]':'(d)', '[CR]':'(e)'}
    num_ys = [col_ys, target_ys]
    for i, df in enumerate([col_df, target_df]):
        # for j, plot in enumerate(plot_list):
        plt.rcParams.update(params)
        # hard coded for two, at the moment
        data_sources = df.source.unique()
        source1_df = df[df['source'] == data_sources[0]]
        source_units1 = config.get_dynamic_attr("{var}.unit", data_sources[0])
        source1_name= data_sources[0]
        source1_id = config.get_dynamic_attr("{var}.data_var", data_sources[0])

        source2_df = df[df['source'] == data_sources[1]]
        source_units2 = config.get_dynamic_attr("{var}.unit", data_sources[1])
        source2_name = data_sources[1]
        source2_id = config.get_dynamic_attr("{var}.data_var", data_sources[1])

        source_var_name = config.get_dynamic_attr("{var}.var_name", data_sources[0])

        if i>0:
            ax = fig.add_subplot(gs[-1, 0], facecolor='none')#, sharex=axs[i-1])
        else:
            ax = fig.add_subplot(gs[i, 0], facecolor='none')


        if i == 0:
            legend_ax = ax

        if i ==0:
            xtick_ax = 'top'
            not_xtick_ax = 'bottom'
            ax.xaxis.set_label_position(xtick_ax)
            ax.xaxis.tick_top()
        else:
            xtick_ax = 'bottom'
            not_xtick_ax = 'top'

        if num_ys[i] !=1:
            plt.style.use('default')
            plt.rcParams.update(params)

            if i ==0:

                if len(plot_list) == 1:
                    legend_ax = ax
            ax2 = ax.twinx()

            # data_sources = df.source.unique()
            # source1_df = df[df['source']==data_sources[0]]
            # source_units1 = config.get_dynamic_attr("{var}.unit", data_sources[0])
            #
            # source2_df = df[df['source']==data_sources[1]]
            # source_units2 = config.get_dynamic_attr("{var}.unit", data_sources[1])

            sns.lineplot(x=time_var, y='value', hue='source', data=source2_df, alpha=alpha, linewidth=linewidth, zorder=-10, ax=ax, palette=pal)
            ax.set_ylabel(source_var_name+'\n'+r'${{{}}}$'.format(source_units2))
            ax.spines['left'].set_color(pal[source2_name])
            ax.tick_params(axis='y', colors=pal[source2_name])
            ax.spines[['right']].set_visible(False)

            sns.lineplot(x=time_var, y='value', hue='source', data=source1_df,alpha=alpha,  zorder=10, linewidth=linewidth,ax=ax2, palette=pal)
            ax2.set_ylabel(r'${{{}}}$'.format(source_units1), rotation=270, labelpad=15)
            ax2.spines['right'].set_color(pal[source1_name])
            ax2.tick_params(axis='y', colors=pal[source1_name])
            ax2.spines[['left']].set_visible(False)
            ax.spines[xtick_ax].set_position(('outward', 20))
            ax2.spines[xtick_ax].set_position(('outward', 20))

            xticks_ts = ax.get_xticks() if xticks_ts is None else xticks_ts
            ax.spines[[xtick_ax]].set_bounds([xticks_ts[1], max([0, xticks_ts[-2]])])
            ax2.spines[[xtick_ax]].set_bounds([xticks_ts[1], max([0, xticks_ts[-2]])])
            ax2.spines[[not_xtick_ax]].set_visible(False)
            ax.spines[[not_xtick_ax]].set_visible(False)


            yticks_ts = ax.get_yticks()
            amplitude = max(np.abs(yticks_ts[0]), np.abs(yticks_ts[-1]))
            ax.set_ylim(-amplitude, amplitude)
            yticks_ts = ax.get_yticks()
            ax.spines[['left']].set_bounds([yticks_ts[1], yticks_ts[-2]])
            ax.set_yticks(yticks_ts[1:-1])

            yticks_ts = ax2.get_yticks()
            amplitude = max(np.abs(yticks_ts[0]), np.abs(yticks_ts[-1]))
            ax2.set_ylim(-amplitude, amplitude)
            yticks_ts = ax2.get_yticks()
            ax2.spines[['right']].set_bounds([yticks_ts[1], yticks_ts[-2]])
            ax2.set_yticks(yticks_ts[1:-1])

            h,l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            ax.get_legend().remove()

            h,l = ax2.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            ax2.get_legend().remove()

            ax.set_xlabel('{}'.format(time_unit), labelpad=15)
            ax2.grid(False)
            ax.grid(False)

            axs.append(ax2)
            axs.append(ax)

            xticks_ts[-1] = max([0, xticks_ts[-2]])
            xticks_ts =xticks_ts[1:]
            ax.spines[[xtick_ax]].set_bounds([xticks_ts[0], max([0, xticks_ts[-2]])])

            ax.set_xticks(ticks = xticks_ts, labels =[f'{np.abs(int(t))}' for t in xticks_ts])
            ax.set_xlabel('{}'.format(time_unit), labelpad=15)
            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax2.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

            loc_code = ''
            if i == 0:
                loc_code_v = 'T'
            else:
                loc_code_v = 'B'

            loc_code_h = 'L'

            loc_code = '[' + loc_code_v + loc_code_h + ']'
            _loc_code = loc_d[loc_code]
            ax.text(x_text_loc, y_text_loc, _loc_code, ha='right', va='bottom',
                    transform=ax.transAxes, size=text_size)

        else:
            plt.style.use('default')
            plt.rcParams.update(params)
            # data_sources = df.source.unique()
            #
            # source1_df = df[df['source']==data_sources[0]]
            #
            # source2_df = df[df['source']==data_sources[1]]
            # source_units2 = config.get_dynamic_attr("{var}.unit", data_sources[1])

            ax = sns.lineplot(x=time_var, y='value', hue='source', data=df, alpha=alpha,  ax=ax, linewidth=linewidth,palette=pal)
            ax.set_ylabel(source_var_name+'\n'+r'${{{}}}$'.format(source_units1))
            ytick_ax = 'left'
            not_ytick_ax = 'right'
            ax.set_xlabel('{}'.format(time_unit), labelpad=15)
            ax.spines[[not_ytick_ax, not_xtick_ax]].set_visible(False)

            yticks_ts = ax.get_yticks()
            amplitude = max(np.abs(yticks_ts[0]), np.abs(yticks_ts[-1]))
            ax.set_ylim(-amplitude, amplitude)

            yticks_ts = ax.get_yticks()
            ax.spines[[ytick_ax]].set_bounds([yticks_ts[1], yticks_ts[-2]])
            ax.set_yticks(yticks_ts[1:-1])
            #
            if xticks_ts is None:
                xticks_ts = ax.get_xticks()
                xticks_ts[-1] = max([0, xticks_ts[-2]])
                xticks_ts = xticks_ts[1:]


            ax.spines[[xtick_ax]].set_bounds([xticks_ts[0], max([0, xticks_ts[-2]])])            # xticks_ts[-1] = max([0, xticks_ts[-2]])
            # xticks_ts =xticks_ts[1:]
            ax.set_xticks(ticks = xticks_ts, labels =[f'{np.abs(int(t))}' for t in xticks_ts])
            ax.set_xlabel('{}'.format(time_unit), labelpad=15)
            # ax.spines[['bottom']].set_bounds([xticks[1], max([0, xticks[-2]])])

            h,l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            ax.get_legend().remove()
            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

            loc_code = ''
            if i ==0:
                loc_code_v='T'
            else:
                loc_code_v='B'

            loc_code_h='L'

            loc_code = '['+loc_code_v+loc_code_h+']'
            _loc_code = loc_d[loc_code]
            ax.text(x_text_loc, y_text_loc, _loc_code, ha='right', va='bottom',
                    transform=ax.transAxes, size=text_size)

            ax.spines[xtick_ax].set_position(('outward', 20))


            axs.append(ax)

        # PSD figs
        if len(plot_list)>1:
            print('plot_list', plot_list)
            plt.style.use('default')
            plt.rcParams.update(params)
            xlims_spec = [5000, 10]
            if i == 0:
                ax_spectral = fig.add_subplot(gs[i, 1], facecolor='none')
                ax_spectral_blocks = fig.add_subplot(gs[1, 1], facecolor='none')
                ax_spectral_blocks.set_yscale('linear')
                ax_spectral_blocks.set_ylim([0, 1])
            else:
                ax_spectral = fig.add_subplot(gs[-1, 1], facecolor='none')


            n = 10000
            for source_name, source_df, source_units, source_id in zip([source1_name, source2_name],
                                                                       [source1_df, source2_df],
                                                                       [source_units1, source_units2],
                                                                       [source1_id, source2_id]):
                source_ps = pyleo.Series(time=-source_df['time'].values, value=source_df['value'].values,
                                         time_unit='yr BP', value_unit=source_units, value_name='TSI',
                                         # label='wu_tsi')
                                         label=source_name)

                psd_source = source_ps.spectral(method='mtm')
                psd_source_signif = psd_source.signif_test(number=n, method='ar1sim', qs=[.95])
                ax_spectral = psd_source_signif.plot(ax=ax_spectral, signif_clr=pal[source_name],
                                                     color=pal[source_name], alpha=0.35, signif_linewidth=1,
                                                     signif_kwargs={'alpha':.6})
                x_scale_spec = ax_spectral.get_xscale()
                ax_spectral_blocks.set_xscale(x_scale_spec)
                freq = psd_source_signif.frequency
                indexes_of_inf = np.argwhere(psd_source_signif.amplitude < psd_source_signif.signif_qs.psd_list[0].amplitude)
                freq_superior = 1/freq
                freq_superior[indexes_of_inf] = np.nan
                amplitude_superior = psd_source_signif.amplitude
                amplitude_superior[indexes_of_inf] = np.nan

                indexes_of_sup = np.argwhere(
                    psd_source_signif.amplitude >= psd_source_signif.signif_qs.psd_list[0].amplitude)
                sup_streaks = find_streaks(indexes_of_sup, delta=2)
                streaks = []
                for streak in sup_streaks:
                    if len(streak) >= 2:
                        streaks.append([streak[0], streak[-1]+1])

                ax_spectral.plot(freq_superior, amplitude_superior, color=pal[source_name], alpha=1)
                # ax_spectral.fill_between(freq_superior, amplitude_superior, 0, where=amplitude_superior > 0,
                #                          color=pal[source_name], alpha=0.3)

                freq_windows = [freq_superior[streak[0]:streak[1]].flatten() for streak in streaks]
                psds_d[source_name] = [[freq_window[0], freq_window[-1]] for freq_window in freq_windows]
                # shaded vertical section
                for streak in streaks:
                    ax_spectral.fill_between(freq_superior[streak[0]:streak[1]], amplitude_superior[streak[0]:streak[1]], 1e-1,
                                             color=pal[source_name], alpha=0.3)

                    ax_spectral_blocks.fill_between(freq_superior[streak[0]:streak[1]],
                                             len(psds_d)/4-.25,len(psds_d)/4,
                                             color=pal[source_name], alpha=0.3)



            plt.style.use('default')

            ax_spectral.set_ylabel('PSD (MTM)')
            ax_spectral.tick_params(axis='y', which='both', labelleft=True)

            yticks = ax_spectral.get_yticks()
            ylims = ax_spectral.get_ylim()

            from matplotlib import colors

            # Rest of your code

            fc = colors.to_rgba('white')
            fc = fc[:-1] + (0.5,)
            bbox = dict(boxstyle="round", fc='w', ec="none", alpha=0.8)


            if i>0:
                ax_spectral.annotate(
                    'de Vries\n(~208 yrs)',
                    xy=(208 + 1, 1e3),
                    xytext=(100 + 1, 1400),
                    ha='left',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=60"
                    ),
                    bbox=bbox
                )

                ax_spectral.annotate(
                    'Hallstatt\n(~2400 yrs)',
                    xy=(2400 + 1, 1e3),
                    xytext=(1400 + 1, 1400),
                    ha='left',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=60"
                    ),
                    bbox=bbox
                )


                ax_spectral.annotate(
                    'Gleissberg\n(~90 yrs)',
                    xy=(88 + 1, 1e2),
                    xytext=(35 + 1, 150),
                    ha='left',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=60"
                    ),
                    bbox=bbox
                )
            cannonical_periods = [88, 208, 2400]
            for period in cannonical_periods:
                ax_spectral.axvline(period, color='k', linestyle=':', alpha=0.35)

            ax_spectral.set_ylim([max(ylims[0],.004), yticks[-2]])
            yticks = ax_spectral.get_yticks()
            ytick_labels = ax_spectral.get_yticklabels()
            ylims = ax_spectral.get_ylim()

            lower_tick_ind = np.argwhere(yticks >= max(ylims[0],1e-1))[0]
            upper_tick_ind = np.argwhere(yticks > ylims[1])[0] - 1
            ytick_ax = 'right'
            not_ytick_ax = 'left'
            ax_spectral.yaxis.tick_right()
            ax_spectral.yaxis.set_label_position(ytick_ax)

            # Hide the left spine and show the right spine
            ax_spectral.spines['left'].set_visible(False)
            ax_spectral.spines[ytick_ax].set_visible(True)

            ax_spectral.spines[[ytick_ax]].set_bounds([yticks[lower_tick_ind[0]], yticks[upper_tick_ind[0]]])
            ax_spectral.set_yticks(ticks=yticks[lower_tick_ind[0]:upper_tick_ind[0] + 1], minor=False)
            ax_spectral.yaxis.set_tick_params(length=0, width=0, color='k', which='minor')

            ax_spectral.set_xlim([5500, 10])
            if i < len(vars) - 1:
                xtick_ax = 'top'
                not_xtick_ax = 'bottom'
                ax_spectral.xaxis.tick_top()
                ax_spectral.xaxis.set_label_position(xtick_ax)

            else:
                xtick_ax = 'bottom'
                not_xtick_ax = 'top'
                legend_ax_spec = ax_spectral
            ax_spectral.spines[[not_xtick_ax]].set_visible(False)

            xticks_spec= ax_spectral.get_xticks()
            xlims_spec = ax_spectral.get_xlim()
            ax_spectral_blocks.set_xlim(xlims_spec)
            ax_spectral_blocks.set_xticks(xticks_spec)

            lower_tick_ind_x = np.argwhere(xticks_spec > xlims_spec[1])[0]
            upper_tick_ind_x = np.argwhere(xticks_spec < xlims_spec[0])[-1]

            ax_spectral.spines[[xtick_ax]].set_bounds(
                [xticks_spec[lower_tick_ind_x[0]], xticks_spec[upper_tick_ind_x[0]] + 1])
            ax_spectral.xaxis.set_tick_params(length=0, width=0, color='k', which='minor')

            ax_spectral.grid(visible=False)
            ax_spectral.legend(frameon=False, loc='lower left')


            ax_spectral.spines[xtick_ax].set_position(('outward', 20))
            ax_spectral.spines[ytick_ax].set_position(('outward', -20))
            xlabel = ax_spectral.get_xlabel()
            ax_spectral.set_xlabel(xlabel, labelpad=15)

            h, l = ax_spectral.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            ax_spectral.get_legend().remove()

            if i ==0:
                loc_code_v='T'
            else:
                loc_code_v='B'

            loc_code_h='R'

            loc_code = '['+loc_code_v+loc_code_h+']'
            _loc_code = loc_d[loc_code]
            ax_spectral.text(x_text_loc, y_text_loc, _loc_code, ha='right', va='bottom',
                    transform=ax_spectral.transAxes, size=text_size)

            axs.append(ax_spectral)

    ax_spectral_blocks.spines['left'].set_visible(False)
    ax_spectral_blocks.spines['right'].set_visible(False)
    ax_spectral_blocks.spines['top'].set_visible(False)
    ax_spectral_blocks.spines['bottom'].set_visible(False)
    ax_spectral_blocks.text(.95, .5, 'Significant\nPeriods', ha='left', va='center',
                     transform=ax_spectral_blocks.transAxes, size=14)
    ax_spectral_blocks.xaxis.set_tick_params(length=0, width=0, color='k', which='minor')
    ax_spectral_blocks.set_xticks([])
    ax_spectral_blocks.set_yticks([])

    _labels = []
    _handles = []
    _extra_labels = []
    _extra_handles = []

    spec_labels = []
    spec_handles = []
    for iq, source in enumerate(labels):
        try:
            label = f'{config.get_dynamic_attr("{var}.var_label", source)}'
            if label not in _labels:
                _labels.append(label)
                _handles.append(handles[iq])
        except:
            label = source
            label = label.replace(', ', ',\n')
            if label not in spec_labels:
                spec_labels.append(label)
                handles[iq].set_color('k')
                spec_handles.append(handles[iq])

    _labels.extend(_extra_labels)
    _handles.extend(_extra_handles)

    fig_target_path = fig_dir / fig_name_noleg
    plt.tight_layout()
    plt.savefig(fig_target_path, dpi=300, bbox_inches='tight')

    legend_ax.legend(_handles, _labels, loc='upper center', bbox_to_anchor=(.5, 0),
                     fontsize=params['legend.fontsize']-1, ncols= 2, frameon=False)

    spec_leg = legend_ax_spec.legend(spec_handles, spec_labels, loc='upper left', bbox_to_anchor=(0, .22),
                          fontsize=params['legend.fontsize'])
    spec_leg.get_frame().set_facecolor('white')
    spec_leg.get_frame().set_edgecolor('white')

    fig_target_path = fig_dir / fig_name_leg
    plt.tight_layout()
    plt.savefig(fig_target_path, dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    # print(f"Number from script2: {args.number}")
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    # with open(proj_dir / 'proj_config.yaml', 'r') as file:
    # config = yaml.safe_load(file)
    config = load_config(proj_dir / 'proj_config.yaml')

    data_dir = proj_dir / config.raw_data.name

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    # may add flag specific things

    if args.flags is not None:
        flags = args.flags
    else:
        flags = ['raw_ts']

    fig_dir = calc_location / config.raw_data.figures_dir
    fig_dir.mkdir(exist_ok=True, parents=True)

    arg_tuple = (config, data_dir, fig_dir, flags)

    plot_raw(arg_tuple)

    # for var, lr in zip([config.target.ids[0], config.col.ids[1]], ['l', 'r']):
    #     embedding_explanation((config, data_dir, fig_dir, flags, var, lr))

    # print("Script2 in verbose mode")
    # if args.parameters is not None:
    #     parameter_flag = args.parameters
    # else:
    #     print('parameters are required', file=sys.stdout, flush=True)
    #     print('parameters are required', file=sys.stderr, flush=True)
    #     sys.exit(0)
