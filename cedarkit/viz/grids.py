import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.plotting.plotting_utils import (
        font_resizer,
        int_yticks_within_ylim,
        replace_supylabel,
        isotope_ylabel,
        replace_latex_labels,
        build_discrete_lag_palette,
    )
    from cedarkit.utils.cli import log_line

except ImportError:
    # Fallback: imports when running as a package
    from utils.plotting.plotting_utils import (
        font_resizer,
        int_yticks_within_ylim,
        replace_supylabel,
        isotope_ylabel,
        replace_latex_labels,
        build_discrete_lag_palette,
    )
    from utils.cli.logging import log_line


class GridCell:
    def __init__(self, row, col, output=None):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.row = row
        self.col = col
        self.occupied = False
        self.row_labels=[]
        self.col_labels=[]
        self.cell_labels=[]
        self.title_labels=[]
        self.output = output
        self.annotations = []
        self.y_lims = []
        annotations = []
        self.relationships = None


class GridPlot:
    def __init__(self, nrows, ncols, width_ratios=None, height_ratios=None, grid_type='plot'):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.nrows = nrows
        self.ncols = ncols
        self.title = None
        self.occupied_dict = {}
        self.ax_grid = {}
        self.ax_grid_types = {}
        self.gridspec_kw = None#{'wspace': 0.07, 'hspace': 0.07} #gridspec_kw={'width_ratios': [2, 1]}
        self.scatter_handles = []
        self.scatter_labels = []
        self.line_handles = []
        self.line_labels = []
        self.fig = None
        self.subfigs = []
        self.ylims = []
        self.xlims = []
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.palette = None
        self.subfigs_d = None
        self.default_ylabel = None
        self.grid_type = grid_type  #'plot' or 'heatmap'
        # self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # self.fig.tight_layout(pad=3.0)

    def set_font_sizes(self, context='paper', multiplier=1.0):
        font_resizer(context=context, multiplier=multiplier)

    def make_grid(self, fig=None, figsize=None, wspace=0.07, hspace=0.07, context='paper', multiplier=1.0):

        self.set_font_sizes(context=context, multiplier=multiplier)

        self.fig = fig if fig is not None else plt.figure(
            figsize=figsize if figsize is not None else (5 * self.ncols, 4 * self.nrows))

        if self.width_ratios is None:
            self.width_ratios = [1 for _ in range(self.ncols)]
        width_ratio_lists = [wr for wr in self.width_ratios if wr is not None and isinstance(wr, (list, tuple))]
        self.subfigs = self.fig.subfigures(self.nrows,  max(1, len(width_ratio_lists)), wspace=wspace, hspace=hspace, height_ratios=self.height_ratios) if self.nrows > 1 else [self.fig]

        subfigs_d = {}
        for row in range(self.nrows):
            if len(width_ratio_lists) == 0:
                subfig = self.subfigs[row] if self.nrows > 1 else self.subfigs[0]

                axes = subfig.subplots(1, self.ncols, gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=self.width_ratios)) if self.ncols > 1 else [subfig.add_subplot(1, 1, 1)]
                if self.ncols == 1:
                    if hspace is not None:
                        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)

                for col in range(self.ncols):
                    self.ax_grid[(row, col, 0)] = axes[col] if self.ncols > 1 else axes[0]
                    self.occupied_dict[(row, col, 0)] = False

            else:
                col_subfigs = subfig.subfigures(1, len(width_ratio_lists), width_ratios=[sum(wr) for wr in width_ratio_lists],
                                                wspace=wspace, hspace=hspace) if len(width_ratio_lists) > 1 else [subfig]
                for ik, width_ratio_list in enumerate(width_ratio_lists):
                    subfigs_d[(row, ik)] = col_subfigs[ik]
                    axes = col_subfigs[ik].subplots(1, len(width_ratio_list), gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=width_ratio_list)) if len(width_ratio_list) > 1 else [col_subfigs[ik].add_subplot(1, 1, 1)]
                    if len(width_ratio_list) == 1:
                        if hspace is not None:
                            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)

                    for jx in range(len(width_ratio_list)):
                        # col = sum([len(wr) for wr in width_ratio_lists[:ik]]) + jx
                        self.ax_grid[(row, jx, ik)] = axes[jx] if len(width_ratio_list) > 1 else axes[0]
                        self.occupied_dict[(row, jx, ik)] = False


        # else:
        #     for row in range(self.nrows):
        #         subfig = self.subfigs[row] if self.nrows > 1 else self.subfigs[0]
        #         col_subfigs = subfig.subfigures(1, self.subfig_cols, width_ratios=[sum(wr) for wr in width_ratio_lists], wspace=wspace, hspace=hspace) if self.subfig_cols > 1 else [subfig]
        #
        #         for ik, width_ratio_list in enumerate(width_ratio_lists):
        #             axes = col_subfigs[ik].subplots(1, len(width_ratio_list), gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=width_ratio_list)) if len(width_ratio_list) > 1 else [col_subfigs[ik].add_subplot(1, 1, 1)]
        #             if len(width_ratio_list) == 1:
        #                 if hspace is not None:
        #                     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)
        #
        #             for jx in range(len(width_ratio_list)):
        #                 # col = sum([len(wr) for wr in width_ratio_lists[:ik]]) + jx
        #                 self.ax_grid[(row, jx, ik)] = axes[jx] if len(width_ratio_list) > 1 else axes[0]
        #                 self.occupied_dict[(row, jx, ik)] = False


    def get_ax(self, row, col, subfig_col=0):
        # ax.set_facecolor('none')
        requested_ax = self.ax_grid.get((row, col, subfig_col), None)
        if requested_ax is not None:
            requested_ax.set_facecolor('none')
        return requested_ax

    def set_ax(self, row, col, ax, subfig_col=0, occupied=True, entry_type='plot'):
        self.ax_grid[(row, col, subfig_col)] = ax
        self.occupied_dict[(row, col, subfig_col)] = occupied#False if (len(ax.lines)==0) and (len(ax.collections)==0) else True
        self.ax_grid_types[(row, col, subfig_col)] = entry_type

    def get_ax_row(self, row):
        return [self.ax_grid.get((row, col, ), None) for col in range(self.ncols)]

    def get_subfig(self, row, col=None):
        if col is None:
            return self.subfigs[row] if self.nrows > 1 else self.subfigs[0]
        else:
            return self.subfigs_d.get((row, col), None)

    def add_handles_labels(self, handles, labels, kind='scatter'):
        if kind == 'scatter':
            for handle, label in zip(handles, labels):
                if label not in self.scatter_labels:
                    self.scatter_handles.append(handle)
                    self.scatter_labels.append(label)
        elif kind == 'line':
            for handle, label in zip(handles, labels):
                if label not in self.line_labels:
                    self.line_handles.append(handle)
                    self.line_labels.append(label)

    def add_annotations(self, ax, add_hline=None):

        ylims = (min(self.ylims), max(self.ylims)) if self.ylims else (None, None)
        if isinstance(add_hline, (int, float)) is True:
            if ylims[0] is None or ylims[1] is None:
                _ylims = ax.get_ylim()
            else:
                _ylims = ylims
            if add_hline>_ylims[0] and add_hline<_ylims[1]:
                ax.axhline(add_hline, color='gray', linestyle='--', linewidth=1)

    def tidy_rows(self, add_hline=None, ylim_by='central', supylabels=None, keep_ylabels=False,
                  supylabel_offset=0.04, keep_titles=False, title_pad=10, rlabel_pad=10, llabel_pad=10, title_rows=[0], titley=1):

        if len(self.subfigs) == 0:
            log_line(logger, "No subfigures available; skipping tidy_rows.", indent=0, log_type="warning")
            return
        if len(self.subfigs[0].axes) == 0:
            log_line(logger, "No axes available; skipping tidy_rows.", indent=0, log_type="warning")
            return

        maxcols = max([col_check_key[1] for col_check_key in self.ax_grid_types.keys()])

        y_tick_list = []
        if ylim_by =='central':
            ylims = (min(self.ylims), max(self.ylims)) if self.ylims else (None, None)
            self.subfigs[0].axes[0].set_ylim(ylims)
            yticks = self.subfigs[0].axes[0].get_yticks()
            delta_y = np.abs(yticks[1] - yticks[0]) if len(yticks) > 1 else 0
            if ylims[0] is not None:
                ylims = [ylims[0] - .25 * delta_y, ylims[1]]

                for ik, subfig in enumerate(self.subfigs):
                    for ip, ax in enumerate(subfig.axes):
                        ax.set_ylim(ylims)

                    y_tick_list.append(yticks)
            # print('got to the ened of central')
            # for ik in range(self.nrows):
            #     y_tick_list.append(yticks)
        #
        # elif ylim_by == 'subfig':
        #     for ik, subfig in enumerate(self.subfigs):
        #         _ylims = []
        #         for im, ax in enumerate(subfig.axes):
        #             if (len(ax.lines) == 0) and (len(ax.collections) == 0):
        #                 continue
        #             n_ylims = ax.get_ylim()
        #             _ylims.append(n_ylims[0])
        #             _ylims.append(n_ylims[1])
        #         _ylims = (min(_ylims), max(_ylims))
        #
        #         if np.abs(_ylims[1]-_ylims[0])>1:
        #             yticks = int_yticks_within_ylim(_ylims[0], _ylims[1])
        #             ylims = (min(min(yticks), _ylims[0])- (yticks[1]-yticks[0])*0.4, _ylims[1]+ (yticks[1]-yticks[0])*0.4)
        #
        #         for ip, ax in enumerate(subfig.axes):
        #             ax.set_ylim(ylims)
        #         y_tick_list.append(yticks)

        elif ylim_by == 'cell':
            # for ik, subfig in enumerate(self.subfigs):
            print('ylim_by cell not implemented yet')


        for ik, subfig in enumerate(self.subfigs):
            ylabel = isotope_ylabel(subfig.axes[0].get_ylabel())
            if ylabel in ['', ' ', None]:
                ylabel = replace_latex_labels(self.default_ylabel)

            supylabel = ''
            if ylabel is not None:
                ylabel_parts = ylabel.rsplit('\n', 1)
                if len(ylabel_parts) > 1:
                    supylabel = replace_latex_labels(replace_supylabel(ylabel_parts[0]))
                    ylabel = '\n'.join(ylabel_parts[1:])
                    if supylabels is not False:
                        if len(supylabel) > 0:
                            supylabels = True

            if supylabels is True:
                subfig.supylabel(supylabel, x=supylabel_offset, va='center', ha='center', fontsize='large',
                                 fontweight='bold')
                subfig.axes[0].set_ylabel(ylabel, rotation=90, labelpad=10, va='center', fontsize='medium')

            subfig_d = {key: self.get_ax(*key) for key in self.ax_grid_types.keys() if key[0] == ik}
            plot_d = {key: ax for key, ax in subfig_d.items() if self.ax_grid_types[key] in ['plot', 'heatmap']}
            max_col = max([key[1] for key in subfig_d.keys()])
            if ylim_by in ['subfig', 'row']:
                _ylims = []
                for key, ax in plot_d.items():
                    n_ylims = ax.get_ylim()
                    _ylims.append(n_ylims[0])
                    _ylims.append(n_ylims[1])
                _ylims = (min(_ylims), max(_ylims))

                if np.abs(_ylims[1] - _ylims[0]) > 1:
                    yticks = int_yticks_within_ylim(_ylims[0], _ylims[1])
                    ylims = (min(min(yticks[1:]), _ylims[0]) - (yticks[1] - yticks[0]) * 0.4,
                             _ylims[-1] + (yticks[1] - yticks[0]) * 0.4)

                for key, ax in plot_d.items():
                    ax.set_ylim(ylims)
                y_tick_list.append(yticks)

            for key, ax in subfig_d.items():
                if (self.ax_grid_types[key] is None) or (self.ax_grid_types[key] =='spacer'): #
                    ax.set_facecolor('none')

                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_title('')

                elif self.ax_grid_types[key] == 'cbar':
                    cbar_ylabel = ax.get_ylabel()
                    ax.set_ylabel(replace_latex_labels(cbar_ylabel), rotation=0, labelpad=10, va='center', fontsize='medium')

                elif self.ax_grid_types[key] in ['legend', 'annotation']:
                    ax.set_facecolor('none')
                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    ax.set_xticks([])

                elif self.ax_grid_types[key] =='title':
                    ax.set_facecolor('none')
                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    ax.set_xticks([])

                    ax.set_title(replace_latex_labels(ax.get_title()), fontsize='large', fontweight='bold', pad=title_pad)

                else:
                    if self.ax_grid_types[key] == 'heatmap':
                        ax.tick_params(axis='y', length=0, width=1)
                        ax.tick_params(axis='x', length=0, width=1)
                        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

                    else:
                        if self.xlims is not None:
                            ax.set_xlim(self.xlims)
                        ax.grid(False)
                        ax.tick_params(axis='y', length=6, width=2)
                        ax.tick_params(axis='x', length=6, width=2)
                        ax.spines['top'].set_visible(False)
                        if add_hline is not None:
                            self.add_annotations(ax, add_hline=add_hline)

                    # tune axis labeling based on content to the right and left
                    left_ax = None
                    try:
                        left_ax_types = [self.ax_grid_types[(key[0], ip, key[2])] for ip in range(key[1])]
                        for iax, atype in enumerate(reversed(left_ax_types)):
                            if atype in ['spacer']:
                                pass
                            elif atype in ['cbar', 'plot', 'annotation', 'legend', 'heatmap']:
                                left_ax = self.get_ax(key[0], key[1] - (iax + 1), key[2])
                                break
                            elif atype is None:
                                break
                        # left_ax = self.ax_grid_types[(key[0], key[1]-1, key[2])]
                    except Exception as e:
                        pass
                    if left_ax is None:
                        if key[1] ==max_col:
                            left_ax = 'forced'

                    right_ax = None
                    try:
                        right_ax_types = [self.ax_grid_types.get((key[0], ip, key[2]), None) for ip in range(key[1]+1, maxcols+1)]
                        for iax, atype in enumerate(right_ax_types):
                            if atype in ['spacer']:
                                pass
                            elif atype in ['cbar', 'plot', 'heatmap']:
                                right_ax = self.get_ax(key[0], key[1] + (iax + 1), key[2])
                                break
                            elif atype is None:
                                break
                    except Exception as e:
                        pass

                    # if there is content in the subplot to the left
                    if (left_ax is not None):
                        ax.spines['left'].set_visible(False)
                        # but there is no content to the right: y-axis on right
                        if right_ax is None:
                            if self.grid_type != 'heatmap':
                                ax.yaxis.tick_right()
                                ax.spines['right'].set_visible(True)
                                ax.spines['right'].set_bounds(yticks[0], yticks[-2])
                                ax.set_yticks(yticks[:-1])
                            else:
                                if (keep_ylabels is False):
                                    ax.set_ylabel('')
                                    ax.set_yticklabels([])
                                    ax.set_yticks([])

                            ylabel = isotope_ylabel(ax.get_ylabel())
                            if supylabel != '':
                                ylabel = ylabel.replace(supylabel, '').strip('\n')
                            ax.set_ylabel(ylabel, rotation=-90, labelpad=rlabel_pad, va='center', fontsize='medium')
                            ax.yaxis.set_label_position("right")

                        # and we don't want ylabels on left in all cases

                        else:
                            ax.spines['right'].set_visible(False)

                            if (keep_ylabels is False):
                                ax.set_ylabel('')
                                ax.set_yticklabels([])
                                ax.set_yticks([])

                    if (left_ax is None) or (keep_ylabels is True):
                        ax.set_ylabel(ylabel, rotation=90, labelpad=llabel_pad, va='center', fontsize='medium')

                        if self.grid_type != 'heatmap':
                            ax.yaxis.tick_left()
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(True)
                            ax.spines['left'].set_bounds(yticks[0], yticks[-2])
                            ax.set_yticks(yticks[:-1])


                    # tune content labeling based on content below
                    below_ax = None
                    try:
                        below_ax = self.ax_grid_types[(key[0] + 1, key[1], key[2])]
                        if below_ax in ['spacer', None, 'cbar', 'annotation', 'legend', 'title']:
                            below_ax = None

                    except Exception as e:
                        pass

                    if len(self.xlims) == 2:
                        # this is probably not needed as it is a check on presence of data which is handled by ax_grid_types
                        if ((len(ax.lines) == 0) and (len(ax.collections) == 0)) is False:
                            ax.set_xlim(self.xlims)

                    if (((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True) or (
                            below_ax is not None):  # (ik < len(self.subfigs) - 1):
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
                        ax.set_xticks([])
                        ax.spines['bottom'].set_visible(False)


                    else:
                        xlabel = ax.get_xlabel()
                        xlabel = xlabel.replace('_', ' ')
                        ax.set_xlabel(xlabel)
                        if (self.xlims is not None) and (len(self.xlims)>1):
                            xticks = int_yticks_within_ylim(self.xlims[0], self.xlims[-1])
                        if self.ax_grid_types[key] =='plot':
                            try:
                                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
                            except:
                                xticks = ax.get_xticks()
                                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])

                    title_text = replace_latex_labels(ax.get_title())
                    if keep_titles == 'individual':
                        ax.set_title(title_text, fontsize='large', fontweight='bold', pad=title_pad)
                    else:
                        if ik in title_rows:
                            ax.set_title(title_text, fontsize='large', fontweight='bold', pad=title_pad)
                        elif (ik > 0) and (supylabels is True):
                            ax.set_title('')
        # for ik, subfig in enumerate(self.subfigs):
        #     ylabel = isotope_ylabel(subfig.axes[0].get_ylabel())
        #     print(ylabel, file=sys.stdout, flush=True)
        #     ylabel_parts = ylabel.rsplit('\n', 1)
        #     supylabel = ''
        #     if len(ylabel_parts) > 1:
        #         supylabel = replace_supylabel(ylabel_parts[0])
        #         ylabel = '\n'.join(ylabel_parts[1:])

            # # column-wise tidy
            # for ip, ax in enumerate(subfig.axes):
            #     ax.grid(False)
            #     ax.tick_params(axis='y', length=5, width=1)
            #     ax.tick_params(axis='x', length=5, width=1)
            #     ax.spines['top'].set_visible(False)
            #     print(self.ax_grid_types[(ik, ip)])
            #
            #     if self.ax_grid_types[(ik, ip)] == 'cbar':
            #         print(self.ax_grid_types[(ik, ip)])#, 'plot'))
            #         continue
            #
            #     if ip > 0:
            #         ax.spines['left'].set_visible(False)
            #         if ip < len(subfig.axes) - 1:
            #             ax.spines['right'].set_visible(False)
            #             ax.set_yticklabels([])
            #             ax.set_yticks([])
            #             ax.set_ylabel('')
            #         else:
            #             if ((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True:
            #                 ax.spines['right'].set_visible(False)
            #                 ax.set_yticklabels([])
            #                 ax.set_yticks([])
            #                 ax.set_ylabel('')
            #             else:
            #                 ax.yaxis.tick_right()
            #                 ax.spines['right'].set_visible(True)
            #
            #                 ylabel = isotope_ylabel(ax.get_ylabel())
            #                 if supylabel != '':
            #                     ylabel = ylabel.replace(supylabel, '').strip('\n')
            #                 ax.set_ylabel(ylabel, rotation=-90, labelpad=25, va='center', fontsize='medium')
            #                 ax.yaxis.set_label_position("right")
            #
            #     else:
            #         if len(supylabel)>0:
            #             subfig.supylabel(supylabel, x=supylabel_offset, va='center', ha='center', fontsize='large', fontweight='bold')
            #             # subfig.axes[ip].set_ylabel(ylabel)
            #
            #         ax.yaxis.tick_left()
            #
            #         ax.spines['left'].set_visible(True)
            #         ax.spines['right'].set_visible(False)
            #         ax.set_ylabel(ylabel, rotation=90, labelpad=20, va='center', fontsize='medium')
            #
            #     if (((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True) or (ik < len(self.subfigs) - 1):
            #         ax.set_xlabel('')
            #         ax.set_xticklabels([])
            #         ax.set_xticks([])
            #         ax.spines['bottom'].set_visible(False)
            #
            #     if ik >0:
            #         if keep_titles is False:
            #             ax.set_title('')
            #     else:
            #         if keep_titles is False:
            #             ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=15)
            #
            #     if len(self.xlims) == 2:
            #         if ((len(ax.lines) == 0) and (len(ax.collections) == 0)) is False:
            #             ax.set_xlim(self.xlims)
            #
            #
            #     yticks = y_tick_list[ik]
            #     subfig.axes[0].spines['left'].set_bounds(yticks[0], yticks[-2])
            #     subfig.axes[-1].spines['right'].set_bounds(yticks[0], yticks[-2])
            #     subfig.axes[0].set_yticks(yticks[:-1])
            #     subfig.axes[-1].set_yticks(yticks[:-1])

        if self.title is not None:
            self.fig.suptitle(replace_latex_labels(self.title), fontsize='x-large', fontweight='bold', y=titley)
        plt.tight_layout()

    def add_legend(self, bbox_to_anchor=(1.05, 1), loc='upper left'):
        handles = self.line_handles + self.scatter_handles
        labels = self.line_labels + self.scatter_labels
        labels = [replace_latex_labels(label) for label in labels]
        ax_legend = self.subfigs[0].axes[-1] if self.nrows > 1 else self.subfigs[0].axes[0]
        if handles:
            ax_legend.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc=loc)



    def _remove_ax(self, ax):
        remove_individually = False
        if ax.get_title() not in ['', ' ', None]:
            remove_individually = 'title'
            # print('title', ax.get_title())
        if ax.get_xlabel() not in ['', ' ', None]:
            remove_individually = 'xlabel'
        if ax.get_ylabel() not in ['', ' ', None]:
            remove_individually = 'ylabel'

        if ax is not None:
            if isinstance(remove_individually, str) is True:
                for loc in ['top', 'right', 'left', 'bottom']:
                    try:
                        ax.spines[loc].set_visible(False)
                    except:
                        pass
                try:
                    ax.grid(False)
                except:
                    pass
                try:
                    ax.set_xticks([])
                except:
                    pass
                try:
                    ax.set_yticks([])
                except:
                    pass

                try:
                    ax.set_xticklabels([])
                except:
                    pass
                try:
                    ax.set_yticklabels([])
                except:
                    pass

                if remove_individually == 'xlabel':
                    ax.set_ylabel('')
                    ax.set_title('')
                elif remove_individually == 'ylabel':
                    ax.set_xlabel('')
                    ax.set_title('')
                elif remove_individually == 'title':
                    ax.set_xlabel('')
                    ax.set_ylabel('')
            else:
                ax.remove()  # ('off')

    def remove_empty(self):
        # subfig_cols = max([col_check_key[2] for col_check_key in self.ax_grid_types.keys()]) + 1
        #
        # maxcols = max([col_check_key[1] for col_check_key in self.ax_grid_types.keys()])

        for key, occupied in self.occupied_dict.items():
            if (occupied is False) or (self.ax_grid_types.get(key, None) in [None, 'spacer']):

                ax = self.get_ax(*key)
                self._remove_ax(ax)

                # if ax.get_title() not in ['', ' ', None]:
                #     remove_individually = 'title'
                #     # print('title', ax.get_title())
                # if ax.get_xlabel() not in ['', ' ', None]:
                #     remove_individually = 'xlabel'
                # if ax.get_ylabel() not in ['', ' ', None]:
                #     remove_individually = 'ylabel'
                #
                # if ax is not None:
                #     if isinstance(remove_individually, str) is True:
                #         for loc in ['top', 'right', 'left', 'bottom']:
                #             try:
                #                 ax.spines[loc].set_visible(False)
                #             except:
                #                 pass
                #         try:
                #             ax.grid(False)
                #         except:
                #             pass
                #         try:
                #             ax.set_xticks([])
                #         except:
                #             pass
                #         try:
                #             ax.set_yticks([])
                #         except:
                #             pass
                #
                #         try:
                #             ax.set_xticklabels([])
                #         except:
                #             pass
                #         try:
                #             ax.set_yticklabels([])
                #         except:
                #             pass
                #
                #         if remove_individually == 'xlabel':
                #             ax.set_ylabel('')
                #             ax.set_title('')
                #         elif remove_individually == 'ylabel':
                #             ax.set_xlabel('')
                #             ax.set_title('')
                #         elif remove_individually == 'title':
                #             ax.set_xlabel('')
                #             ax.set_ylabel('')
                #     else:
                #         ax.remove()#('off')

        self.subfigs = [subfig for subfig in self.subfigs if len(subfig.axes) > 0]


    # def plot_all(self, y_var='delta_rho', palette=None, scatter=False, surr_lines=False, stats_only=True):
    #     for (row, col), (E, tau) in self.location_dict.items():
    #         ax = self.axes[row, col] if self.nrows > 1 and self.ncols > 1 else (self.axes[col] if self.nrows == 1 else self.axes[row])
    #         obj_key = (E, tau)
    #         if obj_key in self.obj_dict:
    #             output = self.obj_dict[obj_key]
    #             lag_plot = LagPlot(y_var=y_var, ax=ax, palette=palette)
    #             lag_plot.make_classic_lag_plot(output)
    #             lag_plot.tidy_plot(legend=False)
    #             ax.set_title(f'E={E}, τ={tau}')
    #         else:
    #             ax.axis('off')  # Turn off axis if no object for this position


class SummaryGrid(GridPlot):
    def __init__(self, nrows, ncols, width_ratios=None, height_ratios=None, grid_type='heatmap'):
        super().__init__(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios, grid_type=grid_type)
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.vlims = []
        self.cbar_ax = None
        self.cbar_label = ''
        self.marker_d = {}
        self.vlims = []
        self.palette = None
        self.sizes= (0, 400)
        self.discrete_lag_mode = False
        self.discrete_lag_values = []
        self.show_corner_legend = False
        self.show_tie_legend = False
        self.show_peak_circle_legend = False
        self.peak_circle_size_samples = [0.2, 0.5, 0.9]
        self.peak_circle_size_range = (30, 380)
        # self.grid_type = 'heatmap'


    def make_colorbar(self):
        self.cbar_ax = self.get_ax(0, self.ncols - 1)
        if self.cbar_ax is None:
            return
        # Ensure we never stack multiple ColorbarBase artists on the same axis.
        self.cbar_ax.cla()

        if self.discrete_lag_mode is True and len(self.discrete_lag_values) > 0:
            lag_vals = sorted([int(v) for v in self.discrete_lag_values])
            lag_info = build_discrete_lag_palette(lag_vals, palette=self.palette)
            n = len(lag_info['lags'])
            cmap = lag_info['cmap']
            norm = lag_info['norm']
            bounds = np.arange(-0.5, n + 0.5, 1)
            mpl.colorbar.ColorbarBase(
                self.cbar_ax,
                cmap=cmap,
                norm=norm,
                boundaries=bounds,
                ticks=np.arange(0, n, 1),
            )
            self.cbar_ax.set_yticklabels([str(v) for v in lag_info['lags']])
            self.cbar_ax.set_ylabel(self.cbar_label if self.cbar_label else 'Optimal lag', labelpad=10)
            return

        norm = mpl.colors.Normalize(vmin=min(self.vlims), vmax=max(self.vlims))
        mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=self.palette, norm=norm)
        self.cbar_ax.set_ylim(self.vlims)
        self.cbar_ax.set_ylabel(self.cbar_label, labelpad=10)

    def create_custom_legend(self, percent_threshold=0.95, bbox_to_anchor = (0,.85)):

        def get_marker_size(value, size_norm, sizes):
            min_norm, max_norm = size_norm
            min_size, max_size = sizes
            norm_value = (value - min_norm) / (max_norm - min_norm)  # Normalize value within size_norm range
            # print('value', value, 'min_size, max_size, norm_values:', min_size, max_size, norm_value)
            return min_size + (max_size - min_size) * norm_value  # Map to sizes range

        # print('labels:', self.line_handles)
        for handle in self.line_handles:
            try:
                handle.set_facecolor('white')  # Ensure marker face is white
                handle.set_edgecolor('.3')
                handle.set_linewidth(.7)
            except Exception as e:
                continue
                print(f"Error setting facecolor for handle {handle}: {e}")

        standard_marker_size = self.line_handles[1].get_markersize()
        nonscaled_marker_size = 15
        # Define your size scaling for `delta_rs_perc_fail`
        size_norm = (1 - percent_threshold, 1)  # The normalization range for size

        # Define sample points for delta_rs_perc_fail to use in the legend
        sample_points = [.4, .25, .1]  # np.linspace(.35, .05, 3)


        # Create size legend elements for `delta_rs_perc_fail`, scaled with base size 20
        _sizes = [get_marker_size(val, size_norm, self.sizes) ** 0.5 for val in sample_points]
        # print('legend sizes:', _sizes)
        # Custom legend for `fill_style` and `diff_from_streak`
        # legend_elements = [
        #     mpl.lines.Line2D([0], [0], marker='o', color='w', label='Temp', markerfacecolor='k', markersize=standard_marker_size),
        #     mpl.lines.Line2D([0], [0], marker='o', color='w', label='TSI', markerfacecolor='black',
        #            markersize=standard_marker_size),
        #     mpl.lines.Line2D([0], [0], marker='o', color='w', label='Both', markerfacecolor='black',
        #            markersize=standard_marker_size),
        #     mpl.lines.Line2D([0], [0], marker='X', color='w', label='Neither', markerfacecolor='black',
        #            markersize=standard_marker_size),
        #     mpl.lines.Line2D([0], [0], marker='s', color='w', label='% Difference from Streak', markerfacecolor='black',
        #            markersize=standard_marker_size)
        # ]

        blank = mpl.lines.Line2D([0], [0], marker='o', color='w', markeredgecolor='w', linewidth=0,
                       markersize=15)
        tmp_l = ['Surrogate type']
        tmp_h = [blank]

        # Combine both legends (size and style)
        for ik, key in enumerate(self.marker_d.keys()):
            if key not in ['statistical', '% deltarho <0', 'end behavior']:
                tmp_h.append(
                    mpl.lines.Line2D([0], [0], marker=self.marker_d[key], color='k', markeredgecolor='w', linewidth=0, label=key,
                           markersize=nonscaled_marker_size))  # standard_marker_size*.6))
                tmp_l.append(key)

        # tmp_h+=[blank, blank]
        # tmp_l+=[' ', 'Additional flags']#.append(' ')

        # for key in marker_d.keys():
        #     if key in ['statistical']:
        #         tmp_h.append(Line2D([0], [0], marker=marker_d[key], color='k', markeredgecolor='w', linewidth=0, label=key, markersize=nonscaled_marker_size))#standard_marker_size*.6))
        #         tmp_l.append(key)
        #
        #     if endbehavior_flag is True:
        #         if key in ['end behavior']:
        #             tmp_h.append(Line2D([0], [0], marker=marker_d[key],
        #                                 # color='orange', markeredgecolor='w', linewidth=0,
        #                                 color='k', markeredgecolor='w', linewidth=0,
        #                                 label=key, markersize=nonscaled_marker_size))#standard_marker_size*.6))
        #             tmp_l.append(key)
        #
        tmp_h += [blank, blank]
        # tmp_l+=[' ', r'$\rho_{\text{final}}$: % surrogate > real']# > $\rho_{\text{final (real)}}$']#+ '\noutperforming ']#.append(' ')
        tmp_l += [' ',
                  '% surrogate ' + r'$\rho_{\text{final}}$' + '\n      > real']  # > $\rho_{\text{final (real)}}$']#+ '\noutperforming ']#.append(' ')

        # tmp_l+=[' ', r'% $\rho_{\text{final (surrogate)}}$ > $\rho_{\text{final (real)}}$']#+ '\noutperforming ']#.append(' ')

        for size in _sizes:
            tmp_h.append(mpl.lines.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', linewidth=0,
                                markersize=size))

        tmp_l.extend([f'{int(val * 100)}%' for val in sample_points])
        tmp_h += [blank, blank]

        yims = self.get_ax(1, 0).get_ylim()
        leg_ax = self.get_ax(1, self.ncols - 1)
        # leg_ax.set_ylim(yims)
        leg_ax.axis('off')
        leg_ax.legend(tmp_h, tmp_l, bbox_to_anchor=bbox_to_anchor,loc='upper left', frameon=False)

    def create_lag_legend(self, bbox_to_anchor=(0, .85)):
        """Legend for lag-grid overlays (base/corner/tie/circle) in discrete lag mode."""
        leg_ax = self.get_ax(1, self.ncols - 1)
        if leg_ax is None:
            return
        leg_ax.axis('off')

        handles = []
        labels = []
        if self.show_corner_legend:
            handles.append(mpl.lines.Line2D([0], [0], marker='>', color='k', markerfacecolor='k', linewidth=0, markersize=10))
            labels.append('Corner: best lag > x')
        if self.show_tie_legend:
            handles.append(mpl.lines.Line2D([0], [0], marker='x', color='k', linewidth=0, markersize=8))
            labels.append('Tie at top rho')
        if self.show_peak_circle_legend:
            min_s, max_s = self.peak_circle_size_range
            for sharp in self.peak_circle_size_samples:
                size = (min_s + (max_s - min_s) * sharp) ** 0.5
                handles.append(mpl.lines.Line2D([0], [0], marker='o', color='k', markerfacecolor='white', linewidth=0, markersize=size))
                labels.append(f'Peak sharpness {sharp:.1f}')

        if len(handles) > 0:
            leg_ax.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='upper left', frameon=False)


    # #@ TODO update GridPlot tidy_rows to handle cbar and spacers
    # def tidy_rows(self, supylabels=True, ylim_by_row=False, supylabel_offset=0.04, titles=False, ylabels_off=True, ):
    #     fall_back_ylab = r'$\tau$'
    #     for ik, subfig in enumerate(self.subfigs):
    #         ylabel = isotope_ylabel(subfig.axes[0].get_ylabel())
    #         if ylabel in ['', ' ', None]:
    #             ylabel = fall_back_ylab
    #
    #         ylabel_parts = ylabel.rsplit('\n', 1)
    #         supylabel = ''
    #         if len(ylabel_parts) > 1:
    #             supylabel = replace_supylabel(ylabel_parts[0])
    #             ylabel = '\n'.join(ylabel_parts[1:])
    #
    #         if supylabels is True:
    #             subfig.supylabel(supylabel, x=supylabel_offset, va='center', ha='center', fontsize='large',
    #                              fontweight='bold')
    #             subfig.axes[0].set_ylabel(ylabel, rotation=90, labelpad=10, va='center', fontsize='medium')
    #
    #         subfig_d = {key: self.get_ax(*key) for key in self.ax_grid_types.keys() if key[0] == ik}
    #         for key, ax in subfig_d.items():
    #             if (self.ax_grid_types[key] is None) or (self.ax_grid_types[key] =='spacer'): #
    #                 ax.set_facecolor('none')
    #
    #                 ax.grid(False)
    #                 ax.tick_params(axis='y', length=0, width=1)
    #                 ax.tick_params(axis='x', length=0, width=1)
    #                 ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    #                 ax.set_yticklabels([])
    #                 ax.set_yticks([])
    #                 ax.set_ylabel('')
    #                 ax.set_xlabel('')
    #                 ax.set_xticklabels([])
    #                 ax.set_xticks([])
    #
    #             elif self.ax_grid_types[key] == 'cbar':
    #                 cbar_ylabel = ax.get_ylabel()
    #                 ax.set_ylabel(cbar_ylabel, rotation=0, labelpad=10, va='center', fontsize='medium')
    #             else:
    #                 ax.tick_params(axis='y', length=0, width=1)
    #                 ax.tick_params(axis='x', length=0, width=1)
    #                 ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    #
    #                 left_ax = None
    #                 try:
    #                     left_ax_types = [self.ax_grid_types[(key[0], ip, key[2])] for ip in range(key[1])]
    #                     for iax, atype in enumerate(reversed(left_ax_types)):
    #                         if atype in ['spacer']:
    #                             pass
    #                         elif atype in ['cbar', 'plot']:
    #                             left_ax = self.get_ax(key[0], key[1]- (iax +1), key[2])
    #                             break
    #                         elif atype is None:
    #                             break
    #                     # left_ax = self.ax_grid_types[(key[0], key[1]-1, key[2])]
    #                 except Exception as e:
    #                     pass
    #
    #                 if (left_ax is not None) and (ylabels_off is True):
    #                         ax.set_ylabel('')
    #                         ax.set_yticklabels([])
    #                         ax.set_yticks([])
    #                 else:
    #                     ax.yaxis.tick_left()
    #                     ax.set_ylabel(ylabel, rotation=90, labelpad=10, va='center', fontsize='medium')
    #
    #                 next_ax = None
    #                 try:
    #                     next_ax = self.ax_grid_types[(key[0]+1, key[1], key[2])]
    #                 except Exception as e:
    #                     pass
    #
    #                 if (((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True) or (next_ax is not None):#(ik < len(self.subfigs) - 1):
    #                     ax.set_xlabel('')
    #                     ax.set_xticklabels([])
    #                     ax.set_xticks([])
    #                     # ax.spines['bottom'].set_visible(False)
    #                 else:
    #                     xlabel = ax.get_xlabel()
    #                     xlabel = xlabel.replace('delta', 'Δ').replace('rho', 'ρ').replace('_', ' ')
    #                     ax.set_xlabel(xlabel)
    #                     xticks = ax.get_xticks()
    #                     ax.get_xticklabels()
    #
    #                 if titles == 'individual':
    #                     ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=10)
    #                 else:
    #                     if (ik > 0) and (supylabels is True):
    #                         ax.set_title('')
    #                         # print('removed title')
    #                     else:
    #                         # print('kept title')
    #                         ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=15)

            # for ip, ax in enumerate(subfig.axes[:-1]):
            #     # ax.grid(False)
            #
            #     if (ik, ip, subfig_col) in self.ax_grid_types.keys() and self.ax_grid_types[(ik, ip,subfig_col )] == 'cbar':
            #         cbar_ylabel = ax.get_ylabel()
            #         ax.set_ylabel(cbar_ylabel, rotation=0, labelpad=10, va='center', fontsize='medium')
            #         continue
            #     if (ik, ip, subfig_col) not in self.ax_grid_types:
            #         ax.set_facecolor('none')
            #     ax.tick_params(axis='y', length=0, width=1)
            #     ax.tick_params(axis='x', length=0, width=1)
            #     ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            #
            #     if ip > 0:
            #         # ax.spines['left'].set_visible(False)
            #         # if ip < len(subfig.axes) - 1:
            #             # ax.spines['right'].set_visible(False)
            #         ax.set_yticklabels([])
            #         ax.set_yticks([])
            #         ax.set_ylabel('')
            #         # else:
            #         #     if ((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True:
            #         #         ax.spines['right'].set_visible(False)
            #         #         ax.set_yticklabels([])
            #         #         ax.set_yticks([])
            #         #         ax.set_ylabel('')
            #         #     else:
            #         #         ax.yaxis.tick_right()
            #         #         ax.spines['right'].set_visible(True)
            #     # else:
            #     #     ylabel = isotope_ylabel(ax.get_ylabel())
            #     #     if supylabel != '':
            #     #         ylabel = ylabel.replace(supylabel, '').strip('\n')
            #     #     ax.set_ylabel(ylabel, rotation=-90, labelpad=25, va='center', fontsize='medium')
            #     #     # ax.yaxis.set_label_position("right")
            #     #
            #     else:
            #
            #         ax.yaxis.tick_left()
            #
            #         # ax.spines['left'].set_visible(False)
            #         # ax.spines['right'].set_visible(False)
            #         # print('ylabel after:', ylabel)
            #
            #     if (((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True) or (ik < len(self.subfigs) - 1):
            #         ax.set_xlabel('')
            #         ax.set_xticklabels([])
            #         ax.set_xticks([])
            #         # ax.spines['bottom'].set_visible(False)
            #     else:
            #         xlabel = ax.get_xlabel()
            #         xlabel = xlabel.replace('delta', 'Δ').replace('rho', 'ρ').replace('_', ' ')
            #         ax.set_xlabel(xlabel)
            #         xticks = ax.get_xticks()
            #         ax.get_xticklabels()
            #
            #     if titles is 'individual':
            #         ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=10)
            #     else:
            #         if (ik >0) and (supylabels is True):
            #             ax.set_title('')
            #             # print('removed title')
            #         else:
            #             # print('kept title')
            #             ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=15)
            # if (ik, ip+1) in self.ax_grid_types and self.ax_grid_types[(ik, ip+1)] == 'cbar':
            #     ax = subfig.axes[-1]
            #     cbar_ylabel = ax.get_ylabel()
            #     ax.set_ylabel(cbar_ylabel, rotation=0, labelpad=10, va='center', fontsize='medium')
            # if (ik, ip+1) not in self.ax_grid_types:
            #     ax = subfig.axes[-1]
            #     ax.set_facecolor('none')
            #     ax.grid(False)
            #     ax.tick_params(axis='y', length=0, width=1)
            #     ax.tick_params(axis='x', length=0, width=1)
            #     ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            #     ax.set_yticklabels([])
            #     ax.set_yticks([])
            #     ax.set_ylabel('')
            #     ax.set_xlabel('')
            #     ax.set_xticklabels([])
            #     ax.set_xticks([])
