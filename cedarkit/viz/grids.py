import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import logging
from collections.abc import Iterable
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.plotting.plotting_utils import (
        font_resizer,
        # int_yticks_within_ylim,
        # replace_supylabel,
        isotope_ylabel,
        replace_latex_labels,
        build_discrete_lag_palette,
    )
    from cedarkit.utils.cli import log_line

except ImportError:
    # Fallback: imports when running as a package
    from utils.plotting.plotting_utils import (
        font_resizer,
        # int_yticks_within_ylim,
        # replace_supylabel,
        isotope_ylabel,
        replace_latex_labels,
        build_discrete_lag_palette,
    )
    from utils.cli.logging import log_line


class GridCell:
    def __init__(self, row, col, output=None, outputs=None):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if output is not None and outputs is not None:
            raise ValueError("Pass either output or outputs, not both.")

        self.row = row
        self.col = col
        self.occupied = False
        self.row_labels=[]
        self.col_labels=[]
        self.cell_labels=[]
        self.title_labels=[]
        self._outputs = []
        if outputs is not None:
            self.outputs = outputs
        elif output is not None:
            self.output = output
        self.annotations = []
        self.y_lims = []
        annotations = []
        self.relationships = None

    @property
    def outputs(self):
        """Ordered collection outputs held by this cell."""
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        if value is None:
            self._outputs = []
            return
        if isinstance(value, (str, bytes)):
            raise TypeError("outputs must be an ordered collection, not a string.")
        self._outputs = list(value)

    @property
    def output(self):
        """Return the legacy singleton output, if the cell has exactly one."""
        if len(self._outputs) == 0:
            return None
        if len(self._outputs) == 1:
            return self._outputs[0]
        raise ValueError("A multi-collection GridCell has no singular output; use outputs.")

    @output.setter
    def output(self, value):
        self._outputs = [] if value is None else [value]

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
        self.xlims = None
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.palette = None
        self.norm = None
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

        if len(width_ratio_lists) > 0:
            tmp_width_ratios = [sum(wr) for wr in width_ratio_lists]
            self.subfigs = self.fig.subfigures(self.nrows, max(1, len(width_ratio_lists)), wspace=wspace, hspace=hspace,
                                               width_ratios=tmp_width_ratios,
                                               height_ratios=self.height_ratios)  # if self.nrows > 1 else [self.fig]


        else:
            tmp_width_ratios = self.width_ratios
            self.subfigs = self.fig.subfigures(self.nrows, max(1, len(width_ratio_lists)), wspace=wspace, hspace=hspace,
                                               # width_ratios=tmp_width_ratios,
                                               height_ratios=self.height_ratios)  # if self.nrows > 1 else [self.fig]

        if isinstance(self.subfigs, Iterable) is False or isinstance(self.subfigs, (str, bytes)):
            self.subfigs = [self.subfigs]

        subfigs_d = {}
        for row in range(self.nrows):
            if len(width_ratio_lists) == 0:
                subfig = self.subfigs[row] if self.nrows > 1 else self.subfigs[0]

                try:
                    axes = subfig.subplots(1, self.ncols, gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=self.width_ratios)) if self.ncols > 1 else [subfig.add_subplot(1, 1, 1)]
                except:
                    axes = subfig[0].subplots(1, self.ncols, gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=self.width_ratios)) if self.ncols > 1 else [subfig[0].add_subplot(1, 1, 1)]
                if self.ncols == 1:
                    if hspace is not None:
                        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)

                for col in range(self.ncols):
                    self.ax_grid[(row, col, 0)] = axes[col] if self.ncols > 1 else axes[0]
                    self.occupied_dict[(row, col, 0)] = False

            else:
                col_subfigs = self.subfigs[row]

                for ik, width_ratio_list in enumerate(width_ratio_lists):
                    subfigs_d[(row, ik)] = col_subfigs[ik]
                    axes = col_subfigs[ik].subplots(1, len(width_ratio_list), gridspec_kw=dict(wspace=wspace, hspace=hspace, width_ratios=width_ratio_list)) if len(width_ratio_list) > 1 else [col_subfigs[ik].add_subplot(1, 1, 1)]
                    if len(width_ratio_list) == 1:
                        if hspace is not None:
                            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)

                    for jx in range(len(width_ratio_list)):
                        self.ax_grid[(row, ik, jx)] = axes[jx] if len(width_ratio_list) > 1 else axes[0]
                        self.occupied_dict[(row, ik, jx)] = False

        self.subfigs_d = subfigs_d

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

    def get_ax_row(self, row, typed_only=False):
        source_keys = self.ax_grid_types.keys() if typed_only is True else self.ax_grid.keys()
        row_keys = sorted([key for key in source_keys if key[0] == row], key=lambda x: (x[2], x[1]))
        return [self.ax_grid.get(key, None) for key in row_keys if self.ax_grid.get(key, None) is not None]

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

    def _validated_xlims(self):
        xlims = self.xlims
        if xlims is None:
            return None
        if not isinstance(xlims, (list, tuple, np.ndarray)):
            return None
        if len(xlims) != 2:
            return None
        left, right = xlims[0], xlims[1]
        try:
            left = float(left)
            right = float(right)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(left) or not np.isfinite(right):
            return None
        return (left, right)

    def _format_latex_label(self, text):
        if not isinstance(text, str):
            return text
        stripped = text.strip()
        if stripped in ['', ' ']:
            return text
        if (stripped.startswith('$$') and stripped.endswith('$$')) or (
            stripped.startswith('$') and stripped.endswith('$') and stripped.count('$') == 2
        ):
            return text
        return replace_latex_labels(text)

    # def collect_plotted_ylims(self, axes):
    #     """Flatten the current (lo, hi) y-limits of a set of axes into one list of bounds."""
    #     values = []
    #     for ax in axes:
    #         if ax is None:
    #             continue
    #         lo, hi = ax.get_ylim()
    #         values.append(lo)
    #         values.append(hi)
    #     return values

    def unify_lims(self, values, mode='centralize', n_ticks=None, decimals=None):
        """Resolve a pool of y-bound values into a single (ylims, yticks) pair."""
        values = [value for value in values if value is not None]
        if len(values) == 0:
            return None, None
        if mode != 'centralize':
            raise ValueError(f"Unknown unify_ylims mode: {mode!r}")

        print('values', values)
        lo, hi = min(values), max(values)
        lo_round, hi_round = lo, hi

        # if np.abs(hi - lo) > 1:
        #     yticks = int_yticks_within_ylim(lo, hi)
        # else:
        if n_ticks is None:
            ticker = mpl.ticker.AutoLocator()
        else:
            ticker = mpl.ticker.MaxNLocator(nbins=n_ticks)


        if (decimals is not None) & ((hi > 0) and (lo < 0)):
            try:
                amp = hi-lo
                tick_delta = amp/(n_ticks-1)
                tick_delta = np.ceil(tick_delta*10**decimals)/10**decimals#, decimals=dec_level)

                if (hi > 0) and (lo < 0):
                    ticks = [0]
                    tick_min = 0
                    while tick_min > lo-.5*tick_delta:
                        tick_min -= tick_delta
                        ticks.append(tick_min)

                    tick_max = 0
                    while tick_max < hi+.5*tick_delta:
                        tick_max += tick_delta
                        ticks.append(tick_max)

                    ticks.sort()
                    rec_ticks = np.array(ticks)
            except:
                pass
        else:
            if decimals is None:
                decimals = np.abs(np.floor(np.log10(np.abs(lo))))
                if np.isnan(decimals):
                    decimals = np.abs(np.floor(np.log10(np.abs(hi))))
                decimals = int(decimals) - 1
            lo_round = np.round(lo, decimals=decimals)
            hi_round = np.round(hi, decimals=decimals)
            rec_ticks = ticker.tick_values(lo, hi)


        # try:
        #     dec_place = np.abs(np.floor(np.log10(np.abs(lo))))
        #     if np.isnan(dec_place):
        #         dec_place = np.abs(np.floor(np.log10(np.abs(hi))))
        #     dec_place = int(dec_place)-1
        #     lo_round = np.round(lo, decimals=dec_place)
        #     hi_round = np.round(hi,decimals= dec_place)
        #     # lo = lo_round
        #     # hi = hi_round
        # except (TypeError, ValueError):
        #     pass

        if decimals is None:
            rec_ticks = ticker.tick_values(lo, hi)
            span = rec_ticks[1] - rec_ticks[0]
            rec_lims = (rec_ticks[0] - span * 0.4, hi_round + span * 0.4)
        else:
            rec_lims = (rec_ticks[rec_ticks<= lo_round][-1], rec_ticks[rec_ticks<= hi_round][-1])
            rec_lims = (min(lo, rec_lims[0]), max(hi, rec_lims[1]))

        rec_ticks = rec_ticks[(rec_ticks >= rec_lims[0]) & (rec_ticks <= rec_lims[1])]
        return rec_lims, rec_ticks

    def tidy_rows(self, add_hline=None, ylim_by='central', supylabels=None, keep_ylabels=False,
                  supylabel_offset=0.04, keep_titles=False, title_pad=10, rlabel_pad=10, llabel_pad=10, title_rows=[0], titley=1,
                  supylabel_target='first', num_xticks=None, xtick_decimals=None, num_yticks=None, ytick_decimals=1, xtick_integer=True):

        if len(self.subfigs) == 0:
            log_line(logger, "No subfigures available; skipping tidy_rows.", indent=0, log_type="warning")
            return

        typed_keys = sorted(self.ax_grid_types.keys(), key=lambda x: (x[0], x[2], x[1]))
        typed_axes = [self.ax_grid.get(key, None) for key in typed_keys if self.ax_grid.get(key, None) is not None]
        if len(typed_axes) == 0:
            log_line(logger, "No axes available; skipping tidy_rows.", indent=0, log_type="warning")
            return

        maxcols = max([col_check_key[1] for col_check_key in typed_keys])
        valid_xlims = self._validated_xlims()

        y_tick_list = []
        row_ylims = {}

        ylims_central = (min(self.ylims), max(self.ylims)) if self.ylims else None
        yticks = np.array([])
        ybounds = None
        if ylim_by == 'central' and ylims_central is not None:
            ylims_central, yticks = self.unify_lims(ylims_central, mode='centralize', n_ticks = num_yticks, decimals=ytick_decimals)
            delta_y = np.abs(yticks[1] - yticks[0]) if len(yticks) > 1 else 0
            if ylims_central is not None:
                ylims_central = [ylims_central[0] - .25 * delta_y, ylims_central[1]]

                for ik in range(self.nrows):
                    row_keys = sorted([key for key in typed_keys if key[0] == ik], key=lambda x: (x[2], x[1]))
                    row_axes = [
                        self.ax_grid[key]
                        for key in row_keys
                        if self.ax_grid.get(key, None) is not None and self.ax_grid_types[key] == 'plot'
                    ]
                    for ax in row_axes:
                        ax.set_ylim(ylims_central)

                    row_ylims[ik] = ylims_central
                    y_tick_list.append(yticks)
            ylims = ylims_central
            ybounds = [yticks[0], yticks[-1]]

        elif ylim_by == 'cell':
            # for ik, subfig in enumerate(self.subfigs):
            print('ylim_by cell not implemented yet')

        for ik in range(self.nrows):
            row_keys = sorted([key for key in typed_keys if key[0] == ik], key=lambda x: (x[2], x[1]))
            if len(row_keys) == 0:
                continue

            row_axes = [self.ax_grid.get(key, None) for key in row_keys if self.ax_grid.get(key, None) is not None]
            if len(row_axes) == 0:
                log_line(logger, f"Row {ik} has typed entries but no axes; skipping row.", indent=0, log_type="warning")
                continue
            row_subfigs = []
            if self.subfigs_d is not None and len(self.subfigs_d) > 0:
                row_subfigs = [self.subfigs_d[key] for key in sorted(self.subfigs_d.keys(), key=lambda x: x[1]) if key[0] == ik]
            if len(row_subfigs) == 0 and len(self.subfigs) > 0:
                row_entry = self.subfigs[ik] if self.nrows > 1 else self.subfigs[0]
                if isinstance(row_entry, Iterable) and not isinstance(row_entry, (str, bytes)):
                    row_subfigs = list(row_entry)
                else:
                    row_subfigs = [row_entry]

            # y axis label
            ylabel = isotope_ylabel(row_axes[0].get_ylabel())
            if ylabel in ['', ' ', None]:
                ylabel = self._format_latex_label(self.default_ylabel)

            # row label
            supylabel = ''
            if ylabel is not None:
                ylabel_parts = ylabel.rsplit('\n', 1)
                if len(ylabel_parts) > 1:
                    supylabel = self._format_latex_label(ylabel_parts[0])
                    ylabel = '\n'.join(ylabel_parts[1:])
                    if supylabels is not False:
                        if len(supylabel) > 0:
                            supylabels = True

            if supylabels is True:
                target_subfigs = []
                if supylabel_target == 'all':
                    target_subfigs = row_subfigs
                elif supylabel_target == 'none':
                    target_subfigs = []
                elif isinstance(supylabel_target, int):
                    if 0 <= supylabel_target < len(row_subfigs):
                        target_subfigs = [row_subfigs[supylabel_target]]
                else:
                    if len(row_subfigs) > 0:
                        target_subfigs = [row_subfigs[0]]

                for subfig in target_subfigs:
                    if hasattr(subfig, 'supylabel'):
                        subfig.supylabel(supylabel, x=supylabel_offset, va='center', ha='center', fontsize='large',
                                         fontweight='bold')
                row_axes[0].set_ylabel(ylabel, rotation=90, labelpad=10, va='center')


            subfig_d = {key: self.ax_grid.get(key, None) for key in row_keys}
            plot_d = {
                key: ax for key, ax in subfig_d.items()
                if (ax is not None) and self.ax_grid_types[key] == 'plot'
            }
            max_col = max([key[1] for key in subfig_d.keys()])
            # yticks = row_axes[0].get_yticks()

            if ylim_by in ['subfig', 'row']:
                plotted_values = [val for ax in plot_d.values() if ax is not None for val in ax.get_ylim()]
                if len(plotted_values) > 0:
                    ylims, yticks = self.unify_lims(plotted_values, mode='centralize', n_ticks=num_yticks, decimals=ytick_decimals)
                    ybounds = [yticks[0], yticks[-1]]

                    for key, ax in plot_d.items():
                        ax.set_ylim(ylims)
                    row_ylims[ik] = ylims
                y_tick_list.append(yticks)

            for key, ax in subfig_d.items():
                if ax is None:
                    continue
                if (self.ax_grid_types[key] is None) or (self.ax_grid_types[key] =='spacer'): #
                    ax.set_facecolor('none')

                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set(yticklabels=[], yticks=[], ylabel='', xlabel='', xticklabels=[], xticks=[], title='')

                elif self.ax_grid_types[key] == 'cbar':
                    cbar_ylabel = ax.get_ylabel()
                    if cbar_ylabel not in ['', ' ', None]:
                        ax.set_ylabel(self._format_latex_label(cbar_ylabel), rotation=0, labelpad=10, va='center', fontsize='medium')

                elif self.ax_grid_types[key] in ['legend', 'annotation']:
                    ax.set_facecolor('none')
                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set(yticklabels= [], yticks=[], ylabel='', xlabel='', xticklabels=[], xticks=[])

                elif self.ax_grid_types[key] =='title':
                    ax.set_facecolor('none')
                    ax.grid(False)
                    ax.tick_params(axis='y', length=0, width=1)
                    ax.tick_params(axis='x', length=0, width=1)
                    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    ax.set(yticklabels=[], yticks=[], ylabel='', xlabel='', xticklabels=[], xticks=[])

                    ax.set_title(self._format_latex_label(ax.get_title()), fontsize='large', fontweight='bold', pad=title_pad)

                else:
                    if self.ax_grid_types[key] == 'heatmap':
                        ax.tick_params(axis='y', length=0, width=1)
                        ax.tick_params(axis='x', length=0, width=1)
                        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                    else:
                        if valid_xlims is not None:
                            ax.set_xlim(valid_xlims)
                        ax.grid(False)
                        ax.tick_params(axis='y', length=6, width=2)
                        ax.tick_params(axis='x', length=6, width=2)
                        ax.spines['top'].set_visible(False)
                        if add_hline is not None:
                            self.add_annotations(ax, add_hline=add_hline)

                    # tune axis labeling based on content to the right and left
                    left_ax = None
                    content_neighbor_types = ['plot', 'heatmap']
                    try:
                        left_ax_types = [self.ax_grid_types[(key[0], ip, key[2])] for ip in range(key[1])]
                        for iax, atype in enumerate(reversed(left_ax_types)):
                            if atype in ['spacer']:
                                pass
                            elif atype in content_neighbor_types:
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
                            elif atype in content_neighbor_types:
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
                                if len(yticks) >= 2:
                                    ax.spines['right'].set_bounds(ybounds[0], ybounds[-1])
                                    ax.set_yticks(yticks)#[:-1])
                            else:
                                if (keep_ylabels is False):
                                    ax.set(ylabel='', yticklabels=[], yticks=[])

                            axis_ylabel = isotope_ylabel(ax.get_ylabel())
                            if axis_ylabel in ['', ' ', None]:
                                axis_ylabel = self._format_latex_label(self.default_ylabel)
                            if (axis_ylabel is not None) and (supylabel != ''):
                                axis_ylabel = axis_ylabel.replace(supylabel, '').strip('\n')
                            axis_ylabel = self._format_latex_label(axis_ylabel)
                            ax.set_ylabel(axis_ylabel, rotation=-90, labelpad=rlabel_pad, va='center')
                            ax.yaxis.set_label_position("right")

                        # and we don't want ylabels on left in all cases

                        else:
                            ax.spines['right'].set_visible(False)

                            if (keep_ylabels is False):
                                ax.set(ylabel='', yticklabels=[], yticks=[])


                    if (left_ax is None) or (keep_ylabels is True):
                        axis_ylabel = isotope_ylabel(ax.get_ylabel())
                        if axis_ylabel in ['', ' ', None]:
                            axis_ylabel = self._format_latex_label(self.default_ylabel)
                        axis_ylabel = self._format_latex_label(axis_ylabel)
                        ax.set_ylabel(axis_ylabel, rotation=90, labelpad=llabel_pad, va='center')

                        if self.grid_type != 'heatmap':
                            ax.yaxis.tick_left()
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(True)
                            if len(yticks) >= 2:
                                ax.spines['left'].set_bounds(ybounds[0], ybounds[-1])
                                ax.set_yticks(yticks)#[:-1])
                            print('not heatmap', yticks)


                    # tune content labeling based on content below
                    below_ax = None
                    try:
                        below_ax = self.ax_grid_types[(key[0] + 1, key[1], key[2])]
                        if below_ax in ['spacer', None, 'cbar', 'annotation', 'legend', 'title']:
                            below_ax = None

                    except Exception as e:
                        pass

                    if valid_xlims is not None:
                        # this is probably not needed as it is a check on presence of data which is handled by ax_grid_types
                        if ((len(ax.lines) == 0) and (len(ax.collections) == 0)) is False:
                            ax.set_xlim(valid_xlims)

                    if (((len(ax.lines) == 0) and (len(ax.collections) == 0)) is True) or (
                            below_ax is not None):  # (ik < len(self.subfigs) - 1):
                        ax.set(xlabel='', xticklabels=[], xticks=[])
                        ax.spines['bottom'].set_visible(False)
                    else:
                        xlabel = ax.get_xlabel()
                        xlabel = xlabel.replace('_', ' ')
                        ax.set_xlabel(xlabel)
                        if valid_xlims is not None:
                            if num_xticks is not None:
                                ax.xaxis.set_major_locator(MaxNLocator(nbins=num_xticks, integer=xtick_integer))
                            xlims, xticks = self.unify_lims(valid_xlims, mode='centralize', n_ticks=num_xticks, decimals=xtick_decimals)
                            # print('xlims_rec', xlims_rec, 'xticks_rec', xticks_rec)
                            # xticks = ax.get_xticks()
                            # xticks = int_yticks_within_ylim(valid_xlims[0], valid_xlims[1])
                            # xticks = xticks[(xticks >= min(valid_xlims)) & (xticks <= max(valid_xlims))]
                            print('xlims',valid_xlims, 'xticks', xticks)

                        if self.ax_grid_types[key] =='plot':
                            try:
                                if len(xticks) >= 2:
                                    ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
                                    ax.set_xticks(xticks)
                                    print('not plot', xticks)
                            except:
                                xticks = ax.get_xticks()
                                tmp_xlims = ax.get_xlim()
                                xlims, xticks = self.unify_lims(tmp_xlims, mode='centralize',
                                                                        n_ticks=num_xticks, decimals=xtick_decimals)
                                # print('xlims_rec', xlims_rec, 'xticks_rec', xticks_rec)
                                if len(xticks) >= 2:
                                    ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
                                    ax.set_xticks(xticks)

                    if self.ax_grid_types[key] == 'plot' and ik in row_ylims:
                        ax.set_ylim(row_ylims[ik])

                    title_text = self._format_latex_label(ax.get_title())
                    if keep_titles is not False:
                        fontweight = 'bold'
                        if isinstance(keep_titles, str) is True:
                            if 'individual' in keep_titles:
                                if 'normal' in keep_titles:
                                    fontweight= 'normal'
                                else:
                                    fontweight = 'bold'

                        ax.set_title(title_text, fontsize='large', fontweight=fontweight, pad=title_pad)
                    else:
                        if ik in title_rows:
                            ax.set_title(title_text, fontsize='large', fontweight='bold', pad=title_pad)
                        elif (ik > 0) and (supylabels is True):
                            ax.set_title('')

        if self.title is not None:
            self.fig.suptitle(self._format_latex_label(self.title), fontsize='x-large', fontweight='bold', y=titley)
        plt.tight_layout()

    def add_legend(self, bbox_to_anchor=(1.05, 1), loc='upper left'):
        handles = self.line_handles + self.scatter_handles
        labels = self.line_labels + self.scatter_labels
        labels = [self._format_latex_label(label) for label in labels]
        row_axes = self.get_ax_row(0, typed_only=False)
        if handles and len(row_axes) > 0:
            ax_legend = row_axes[-1]
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

        def _has_axes(entry):
            if hasattr(entry, 'axes'):
                try:
                    return len(entry.axes) > 0
                except Exception:
                    return False
            if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
                for subentry in entry:
                    if _has_axes(subentry):
                        return True
            return False

        self.subfigs = [subfig for subfig in self.subfigs if _has_axes(subfig)]


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
        # print('norm', self.norm)
        active_axes = set(self.ax_grid.values())
        if self.cbar_ax not in active_axes:
            self.cbar_ax = None

        if self.cbar_ax is None:
            cbar_keys = [key for key, entry_type in self.ax_grid_types.items() if entry_type == 'cbar']
            if len(cbar_keys) > 0:
                self.cbar_ax = self.ax_grid.get(cbar_keys[0], None)
        if self.cbar_ax is None:
            self.cbar_ax = self.get_ax(0, self.ncols - 1)
        if self.cbar_ax is None:
            return

        # Ensure we never stack multiple ColorbarBase artists on the same axis.
        self.cbar_ax.cla()
        cmap = self.palette
        if cmap is None:
            return
        if isinstance(cmap, str):
            cmap = mpl.colormaps.get_cmap(cmap)
        elif not isinstance(cmap, mpl.colors.Colormap):
            cmap = mpl.colors.LinearSegmentedColormap.from_list('summary_grid_palette', list(cmap))

        if self.discrete_lag_mode is True and len(self.discrete_lag_values) > 0:
            lag_vals = sorted([int(v) for v in self.discrete_lag_values])
            lag_info = build_discrete_lag_palette(lag_vals, palette=cmap)
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

        if self.norm is not None:
            norm = self.norm
        else:
            has_vlims = isinstance(self.vlims, (list, tuple, np.ndarray)) and len(self.vlims) == 2
            if has_vlims:
                norm = mpl.colors.Normalize(vmin=self.vlims[0], vmax=self.vlims[1])
            elif isinstance(self.vlims, (list, tuple, np.ndarray)) and len(self.vlims) > 0:
                norm = mpl.colors.Normalize(vmin=min(self.vlims), vmax=max(self.vlims))
            else:
                print('No valid vlims provided for colorbar; skipping colorbar creation.')
                return

        mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=cmap, norm=norm)
        norm_vmin = getattr(norm, 'vmin', None)
        norm_vmax = getattr(norm, 'vmax', None)
        if norm_vmin is not None and norm_vmax is not None:
            self.vlims = (norm_vmin, norm_vmax)
            self.cbar_ax.set_ylim([norm_vmin, norm_vmax])
        elif isinstance(self.vlims, (list, tuple, np.ndarray)) and len(self.vlims) == 2:
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
