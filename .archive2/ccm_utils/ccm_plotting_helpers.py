import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

import cedarkit.utils.routing.paths


def clear_ax_annotation(_ax):
    for loc in ['left', 'right', 'top', 'bottom']:
        _ax.spines[loc].set_visible(False)

    _ax.set_xticklabels([])
    _ax.set_xticks([])
    _ax.set_yticklabels([])
    _ax.set_yticks([])
    _ax.grid(False)
    _ax.set_facecolor('none')
    return _ax

def make_lag_subplots_ccm(plot_lags, _plot_df, alpha=.6, title_copy=None, nbins=None, label_d=None, cmap='viridis', split_relation=False, color_palette=None, **kwargs):
    tau_string = r'$\tau$'
    rho_string = r'$\rho$'

    def set_alpha(cmap_name, alpha):
        base_cmap = plt.get_cmap(cmap_name)
        colors = base_cmap(np.arange(base_cmap.N))
        alphas = np.ones((colors.shape[0], 1)) * alpha
        new_colors = np.concatenate((colors[:, :3], alphas), axis=1)
        return mcolors.ListedColormap(new_colors)
    def slim_xlabels(label, label_d):
        label = cedarkit.utils.routing.paths.replace('_noise', '')
        for key in label_d.keys():
            label = cedarkit.utils.routing.paths.replace(key, label_d[key])
        return label

    # E = plot_df.E.unique()[0]
    # Tp_max = plot_df.Tp.unique()[0]
    # tau = plot_df.tau.unique()[0]
    fig_obj = []
    for traits, plot_df in _plot_df.groupby(['E', 'Tp', 'tau']):
        E = traits[0]
        Tp_max = traits[1]
        tau = traits[2]

        plot_df = plot_df.copy()
        relations = plot_df.relation.unique()
        title_copy = title_copy if title_copy is not None else ''
        label_d = label_d if label_d is not None else {}

        if nbins is None:
            nbins = int(.5*(plot_df.LibSize.max()-plot_df.LibSize.min()))

        bin_edges = list(np.linspace(plot_df.LibSize.min(), plot_df.LibSize.max() + 1, nbins, dtype=int))
        plot_df['lib_bin'] = pd.cut(plot_df['LibSize'], bins=bin_edges, include_lowest=True)

        min_max = []
        for ind, grp in plot_df.groupby(['lag', 'lib_bin', 'relation'], observed=True):
            bin_edges = list(np.linspace(grp.rho.min(), grp.rho.max(), 20))
            grp['rho_bin'] = pd.cut(grp['rho'], bins=bin_edges, include_lowest=True)

            min_max += [len(grp) for ind, grp in grp.groupby(['rho_bin'], observed=True)]
        norm1 = Normalize(vmin=min(min_max), vmax=max(min_max))

        # reorganize plot lags
        neg_half = plot_lags[plot_lags <= 0]
        pos_half = np.sort(plot_lags[plot_lags > 0])[::-1]
        plot_lags = np.concatenate([neg_half, pos_half])

        # Determine the figure layout
        n_rows = 2 if not split_relation else 5
        height_ratios = [1] * n_rows
        if split_relation:
            # for i in range(1, n_rows, 3):
            height_ratios[2] = .2

        fig = plt.figure(figsize=(20, 7 * (2 if split_relation else 1)+(2 if split_relation else 0)))
        gs = GridSpec(n_rows, len(neg_half), figure=fig, height_ratios=height_ratios)

        # Adjust axs creation based on whether we're splitting by relation
        axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(len(neg_half))]

        if split_relation:
            axs += [fig.add_subplot(gs[i, j]) for i in range(3,n_rows) for j in range(len(neg_half))]
            colorbar_gs = [GridSpecFromSubplotSpec(20, 10, subplot_spec=gs[1, -1]),
                       GridSpecFromSubplotSpec(20, 10, subplot_spec=gs[-1, -1])]  # Nested gridspec for colorbars]
        else:
            colorbar_gs = [GridSpecFromSubplotSpec(20, 10, subplot_spec=gs[-1, -1])]  # Nested gridspec for colorbars]

        title_copy = '\n'.join([title_copy, f'Tp={Tp_max}, E={E}, {tau_string}={tau}'])
        title_copy = title_copy.lstrip('\n')

        plot_shift = len(neg_half) - 1 - len(pos_half)
        for ind, lag in enumerate(plot_lags):
            if ind > len(neg_half) - 1:
                ind += plot_shift

            legend = False
            if ind == len(plot_lags) - 1:
                legend = True

            grp_df = plot_df[plot_df['lag'] == lag]
            if split_relation:
                for rel_ind, relation in enumerate(relations):
                    ax = axs[(rel_ind+ind) + rel_ind * len(plot_lags)]
                    rel_df = grp_df[grp_df['relation'] == relation]
                    sns.histplot(x="LibSize", y="rho", hue='relation', cbar=False, data=rel_df[rel_df['Tp'] == Tp_max],
                                 kde=True, ax=ax, common_norm=norm1, legend=legend, palette={relation: color_palette.get(relation, sns.color_palette("Set2")[rel_ind]) if color_palette else sns.color_palette("Set2")[rel_ind]}, alpha=alpha,
                                 bins=nbins, **kwargs)
                    ax.set_title(f'lag={lag}')
                    if ind <= len(pos_half) - 1:
                        ax.set_xlabel(None)
                        ax.set_xticklabels([])

                    if ind not in [len(neg_half), 0]:
                        ax.set_ylabel(None)
                        ax.set_yticklabels([])
                    else:
                        ylabel = ax.get_ylabel()
                        ax.set_ylabel(cedarkit.utils.routing.paths.replace('rho', rho_string))

                    ax.set_ylim([-.25, 1])

                    if legend:
                        color_maps = {}
                        labels = ax.legend_.texts
                        handles = ax.legend_.legend_handles
                        for ip, label in enumerate(labels):
                            label = label.get_text()
                            color_maps[label] = set_alpha(_cmap_from_color(handles[ip]._facecolor), alpha)
                        ax.legend().remove()

                        cb_axs = []
                        for ip, label in enumerate(color_maps.keys()):
                            cb_ax = fig.add_subplot(colorbar_gs[rel_ind][4:-2, ip*2])
                            cb_axs.append(cb_ax)

                            plt.colorbar(plt.cm.ScalarMappable(norm=norm1, cmap=color_maps[label]),
                                         cax=cb_axs[ip])  # , label=slim_xlabels(key, label_d))

                            cb_axs[ip].set_title(slim_xlabels(label, label_d), loc='left')
                            # cb_axs[ip].set_yticklabels([])

                        ax_clear = axs[(rel_ind+ind) + rel_ind * len(plot_lags)+1]
                        ax_clear = clear_ax_annotation(ax_clear)

                        plt.suptitle(title_copy, y=.95)

            else:
                ax = axs[ind]
                sns.histplot(x="LibSize", y="rho", hue='relation', cbar=False, data=grp_df[grp_df['Tp'] == Tp_max],
                             kde=True, ax=ax, common_norm=norm1, legend=legend,
                             palette=color_palette if color_palette else sns.color_palette("Set2"), alpha=alpha,
                             bins=nbins, **kwargs)

                ax.set_title(f'lag={lag}')
                if ind <= len(pos_half) - 1:
                    ax.set_xlabel(None)
                    ax.set_xticklabels([])

                if ind not in [len(neg_half), 0]:
                    ax.set_ylabel(None)
                    ax.set_yticklabels([])
                else:
                    ylabel = ax.get_ylabel()
                    ax.set_ylabel(cedarkit.utils.routing.paths.replace('rho', rho_string))

                ax.set_ylim([-.25, 1])

                if legend:
                    color_maps = {}
                    labels = ax.legend_.texts
                    handles = ax.legend_.legend_handles
                    for ip, label in enumerate(labels):
                        label = label.get_text()
                        color_maps[label]=set_alpha(_cmap_from_color(handles[ip]._facecolor), alpha)
                    ax.legend().remove()

                    cb_axs = []
                    color_map_keys= list(color_maps.keys())
                    for ip, label in enumerate(color_map_keys):
                        cb_ax = fig.add_subplot(colorbar_gs[0][4:-2, ip * 2])
                        cb_axs.append(cb_ax)

                        plt.colorbar(plt.cm.ScalarMappable(norm=norm1, cmap=color_maps[label]),
                                     cax=cb_axs[ip])  # , label=slim_xlabels(key, label_d))


                    xlabel2 = color_map_keys[1]  # .replace('causes', 'causes\n')
                    cb_axs[1].set_xlabel(slim_xlabels(xlabel2, label_d), loc='left')
                    fontsize2 = cb_axs[1].xaxis.label.get_fontsize()

                    cb_axs[0].set_title(slim_xlabels(color_map_keys[0], label_d), loc='left', fontsize=fontsize2)
                    cb_axs[0].set_yticklabels([])

                    ax_clear = axs[ind + 1]
                    ax_clear = clear_ax_annotation(ax_clear)

                    plt.suptitle(title_copy, y=1.05)

        fig_obj.append((fig, axs))
    if len(fig_obj) == 1:
        return fig, axs
    else:
        return fig_obj
    # return fig, axs


# from seaborn
from matplotlib.colors import to_rgba
import numpy as np
import matplotlib as mpl
from . import husl


def cmap_from_color(color):
    """Return a sequential colormap given a color seed."""
    r, g, b, _ = to_rgba(color)
    h, s, _ = husl.rgb_to_husl(r, g, b)
    xx = np.linspace(-1, 1, int(1.15 * 256))[:256]
    ramp = np.zeros((256, 3))
    ramp[:, 0] = h
    ramp[:, 1] = s * np.cos(xx)
    ramp[:, 2] = np.linspace(35, 80, 256)
    colors = np.clip([husl.husl_to_rgb(*hsl) for hsl in ramp], 0, 1)
    return mpl.colors.ListedColormap(colors[::-1])


