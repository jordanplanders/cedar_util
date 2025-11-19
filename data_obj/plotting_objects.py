import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import compute as pc
import pyarrow as pa
import re
import matplotlib as mpl
import math
import numpy as np
import sys
from matplotlib.markers import MarkerStyle

from copy import copy

def font_resizer(context='paper', multiplier=1.0):
    if context == 'paper':
        sns.set_context("paper", rc={
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16
        })
    elif context == 'talk':
        sns.set_context("talk", rc={
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20
        })
    elif context == 'poster':
        sns.set_context("poster", rc={
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 24
        })
    else:
        sns.set_context("notebook")  # default

    if multiplier != 1.0:
        for key in mpl.rcParams.keys():
            if 'size' in key and isinstance(mpl.rcParams[key], (int, float)):
                mpl.rcParams[key] *= multiplier

        sns.set_context(rc=mpl.rcParams)

def check_palette_syntax(palette, table):
    relation_col = 'relation'
    if relation_col not in table.schema.names:
        relation_col = 'relation_0' if 'relation_0' in table.schema.names else None
    relations = pc.unique(table[relation_col]).to_pylist()
    rel_word = 'causes' if any('cause' in r for r in relations) else 'influences'
    palette_rel_word = 'causes' if any('cause' in r for r in palette.keys()) else 'influences'
    # new_palette = {}
    # for k, v in palette.items():
    #     new_key = k.replace(palette_rel_word, rel_word)
    #     print(f"Replacing palette key '{k}' with '{new_key}'")
    #     new_palette[new_key] = v
    palette = {k.replace(palette_rel_word, rel_word): v for k, v in palette.items()}
    return palette

_SEPS = [r"\s*->\s*", r"\s*→\s*", r"\s*=>\s*", r"\s+causes\s+", r"\s+influences\s+"]

def _parse_relation_once(rel: str) -> tuple[str, str] | None:
    for sep in _SEPS:
        m = re.split(sep, rel.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(m) == 2:
            a, b = m[0].strip(), m[1].strip()
            if a and b:
                return a, b
    # fallback regex (“A causes B” or “A influences B”)
    m = re.match(r"^\s*(.*?)\s+(causes|influences)\s+(.*?)\s*$", rel, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(3).strip()
    return None

def infer_var_names_from_relation(table: pa.Table, relation_col: str = "relation") -> tuple[str, str]:

    if relation_col not in table.schema.names:
        raise KeyError(f"Missing column: {relation_col}")
    # get uniques without materializing full column
    # enc = pc.unique(table[relation_col]).to_pylist() #pc.dictionary_encode(table[relation_col])
    uniques = pc.unique(table[relation_col]).to_pylist()
    names = set()
    for r in uniques:
        parsed = _parse_relation_once(r)
        if parsed:
            names.update(parsed)
    if len(names) != 2:
        raise ValueError(f"Could not infer exactly two variable names from relations; found: {sorted(names)}")
    a, b = sorted(names)  # order doesn’t matter for labeling; pick a stable order
    return a, b

def add_relation_s_inferred(
        table: pa.Table,
        x_var_name: str = None,
        y_var_name: str = None,
        surr_col: str = "surr_var",
        relation_col: str = "relation_0",
) -> pa.Table:

    # print('table schema names', table.schema.names)
    if relation_col not in table.schema.names:
        relation_col = "relation"
    if relation_col not in table.schema.names or surr_col not in table.schema.names:
        raise KeyError(f"Need columns '{relation_col}' and '{surr_col}'")
    x_var_name, y_var_name = infer_var_names_from_relation(table, relation_col)
    # print(f"Inferred variable names: '{x_var_name}', '{y_var_name}'")
    table = table.combine_chunks()

    rel = table[relation_col]
    surr = table[surr_col]

    # Masks
    m_neither = pc.equal(surr, "neither")
    m_both = pc.equal(surr, "both")
    m_x = pc.equal(surr, x_var_name)
    m_y = pc.equal(surr, y_var_name)

    # Variants
    rel_x = pc.replace_substring(rel, x_var_name, f"{x_var_name} (surr) ")
    rel_y = pc.replace_substring(rel, y_var_name, f"{y_var_name} (surr) ")
    rel_both = pc.replace_substring(rel_x, y_var_name, f"{y_var_name} (surr) ")

    # 2) Use nested if_else instead of case_when (robust with chunked/contiguous)
    rel_s = pc.if_else(
        m_neither, rel,
        pc.if_else(
            m_both, rel_both,
            pc.if_else(
                m_x, rel_x,
                pc.if_else(m_y, rel_y, rel)
            )
        )
    )
    rel_s = pc.replace_substring(rel_s, "  ", " ")#.str.lstrip().str.rstrip()
    rel_s = pc.ascii_trim(rel_s, ' ')

    # table.append_column(f"{relation_col}_0", rel)

    # Rename original relation -> relation_0, then insert new relation next to it
    cols = [ f"{c}_0" if (c =='relation') and (relation_col=='relation')  else c for c in table.schema.names]
    # print('end', cols)
    table = table.rename_columns(cols)
    # i0 = table.schema.get_field_index(f"{relation_col}_0")
    table = table.append_column(relation_col, rel_s)
    # print('after append col', table.schema.names)
    # table[relation_col] = pc.ascii_trim(table[relation_col], ' ')
    return table

def int_yticks_within_ylim(ymin, ymax):
    # Find all integer values within the current limits
    ticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    # Ensure at least 2 ticks (for degenerate ranges)
    if len(ticks) < 2:
        ticks = np.array([np.floor(ymin), np.ceil(ymax)])
    return ticks.astype(int)

def replace_supylabel(label):
    label = label.replace('Doering', 'Döring')
    return label

def int_yticks_from_ylim(ymin, ymax):
    # Ensure ymin < ymax
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    # Compute rough range and ideal tick spacing
    yrange = ymax - ymin
    rough_spacing = yrange / 2  # aim for ~3 ticks total (2 intervals)

    # Round spacing to nearest "nice" integer (1, 2, 5, 10, etc.)
    exp = math.floor(math.log10(rough_spacing))
    base = rough_spacing / (10 ** exp)
    if base < 1.5:
        nice_base = 1
    elif base < 3.5:
        nice_base = 1
    elif base < 7.5:
        nice_base = 5
    else:
        nice_base = 10
    spacing = nice_base * (10 ** exp)

    # Compute tick positions
    tick_start = math.floor(ymin / spacing) * spacing
    tick_end = math.ceil(ymax / spacing) * spacing
    ticks = np.arange(tick_start-spacing, tick_end + spacing, spacing)

    # Ensure at least 2 ticks
    if len(ticks) < 2:
        ticks = np.array([math.floor(ymin), math.ceil(ymax)])
    elif len(ticks) == 2:
        # Try to add a middle tick if possible
        mid = np.mean(ticks)
        if mid.is_integer():
            ticks = np.array([ticks[0], mid, ticks[1]])

    return ticks.astype(int)
# dictionary mapping location (row, col) to key
# dictionary mapping key to Output object

#wrap grid?
# subfigure rows

# E x tau grid: {(E, tau): Object}, Location grid {(row, col): (E, tau)}
# E_rows = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7}
# tau_cols = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5}
#
def isotope_ylabel(isotope):
    isotope_labels = {
        'd18O': r'$\delta^{18}O$',
        'dD': r'$\delta D$',
        'd_excess': r'$d$-excess',
        'deltaT': r'$\Delta T$',
        'tanom': r'Temp Anomaly',
        # 'tsi_anom': r'TSI Anomaly (W/m²)',
    }
    for key in isotope_labels.keys():
        if key in isotope:
            isotope = isotope.replace(key, isotope_labels[key])
    return isotope

class GridCell:
    def __init__(self, row, col, output=None):
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
                  supylabel_offset=0.04, keep_titles=False, title_pad=10, rlabel_pad=10, llabel_pad=10, title_rows=[0]):

        maxcols = max([col_check_key[1] for col_check_key in self.ax_grid_types.keys()])

        y_tick_list = []
        if ylim_by =='central':
            ylims = (min(self.ylims), max(self.ylims)) if self.ylims else (None, None)
            self.subfigs[0].axes[0].set_ylim(ylims)
            yticks = self.subfigs[0].axes[0].get_yticks()
            delta_y = np.abs(yticks[1] - yticks[0])
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
                ylabel = self.default_ylabel

            supylabel = ''
            if ylabel is not None:
                ylabel_parts = ylabel.rsplit('\n', 1)
                if len(ylabel_parts) > 1:
                    supylabel = replace_supylabel(ylabel_parts[0])
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
                    ax.set_ylabel(cbar_ylabel, rotation=0, labelpad=10, va='center', fontsize='medium')

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

                    ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=title_pad)

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
                        xlabel = xlabel.replace('delta', 'Δ').replace('rho', 'ρ').replace('_', ' ')
                        ax.set_xlabel(xlabel)
                        if (self.xlims is not None) and (len(self.xlims)>1):
                            xticks = int_yticks_within_ylim(self.xlims[0], self.xlims[-1])
                        if self.ax_grid_types[key] =='plot':
                            try:
                                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
                            except:
                                xticks = ax.get_xticks()
                                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])

                    if keep_titles == 'individual':
                        ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=title_pad)
                    else:
                        if ik in title_rows:
                            ax.set_title(ax.get_title(), fontsize='large', fontweight='bold', pad=title_pad)
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


        plt.tight_layout()

    def add_legend(self, bbox_to_anchor=(1.05, 1), loc='upper left'):
        handles = self.line_handles + self.scatter_handles
        labels = self.line_labels + self.scatter_labels
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



class BasePlot:
    """Class to create lag plots with optional scatter and highlighted points.
    Parameters
    ----------
    y_var : str, default 'delta_rho'
        The y-axis variable to plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    palette : dict or seaborn-compatible palette, optional
        Color palette for different relation categories.

    Methods
    -------
    add_scatter(df, hue='relation', legend=True)
        Adds scatter points to the plot.
    highlight_points(df, hue='relation', edgecolor="black", legend=False)
        Highlights specific points on the plot.
    add_line(df, hue='relation', units='surr_num', legend=False)
        Adds line plots to the plot.
    make_lag_plot(output, scatter=False, surr_lines=False, stats_only=True)
        Creates the lag plot with options for scatter and surrogate lines.
    Attributes
    ----------
    top_val_color : str
        Color for highlighting top values.
    bottom_val_color : str
        Color for highlighting bottom values.
    highlight_points_size : int
        Size of highlighted points.
    highlight_points_linewidth : float
        Line width of highlighted points.
    highlight_points_alpha : float
        Alpha transparency of highlighted points.
    scatter_points_size : int
        Size of scatter points.
    scatter_points_alpha : float
        Alpha transparency of scatter points.

    Examples
    --------
    >>> lag_plot = LagPlot(y_var='delta_rho', palette=my_palette)
    >>> lag_plot.make_lag_plot(output=my_output, scatter=True, surr_lines=True, stats_only=False)

    """


    def __init__(self, grp_d):
        self.y_var = None #y_var
        self.x_var = None # x_var
        self.palette = None #palette
        # self.top_val_color = 'black'
        # self.bottom_val_color = 'gray'
        # self.highlight_points_size = 40
        # self.highlight_points_linewidth = 1.5
        # self.highlight_points_alpha = 1
        self.scatter_points_size = 20
        self.scatter_points_alpha = 0.5

        self.scatter_handles = []
        self.scatter_labels = []

        self.line_handles = []
        self.line_labels = []

        self.min_y = None
        self.max_y = None

        self.annotations = []
        self.ax = None
        self.relation_scope_real=None
        self.relation_scope_surr=None

        if grp_d is not None:
            self.populate(grp_d)

    def populate(self, grp_d):
        for key, value in grp_d.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.ax is None:
            self.ax = plt.subplots(figsize=(8, 6))[1]

    def pull_df(self, output, columns= None):
        return output.select(columns).to_pandas()


    def handle_legend(self, collect_legend=True, legend=False, element_type='scatter'):
        if collect_legend is True:
            handles, labels = self.ax.get_legend_handles_labels()
            # print('handles, labels', handles, labels)
            if element_type == 'scatter':
                for ik in range(len(handles)):
                    label=labels[-(ik+1)]
                    # print('label', label)
                    handle=handles[-(ik+1)]
                    # print(type(handle))
            # for handle, label in zip(handles, labels):

                    if isinstance(handle, (mpl.lines.Line2D) ) is False:
                        if label not in self.scatter_labels:
                            self.scatter_handles.append(handle)
                            self.scatter_labels.append(label)
                            # print('added scatter handle/label', handle, label)
            elif element_type == 'line':
                for handle, label in zip(handles, labels):
                    if label not in self.line_labels:
                        self.line_handles.append(handle)
                        self.line_labels.append(label)
                        #             self.line_handles.append(handle)
                        #             self.line_labels.append(label)

        if legend is False:
            self.ax.legend().remove()

    def tidy_plot(self, legend=False, edge=True, bottom=True):
        # Axis labels
        # self.ax.set_xlabel(self.x_var)
        self.ax.set_ylabel(self.y_var.replace('delta_', 'Δ').replace('rho_', 'ρ'))

        available_ylabel = self.ax.get_ylabel()
        available_ylabel = available_ylabel.replace('rho', "ρ").replace('delta', "Δ").replace('_', ' ')
        self.ax.set_ylabel(available_ylabel)
        # Remove duplicate legend entries if scatter used
        self.ax.grid(False)
        self.ax.tick_params(axis='y', length=5, width=1)
        self.ax.tick_params(axis='x', length=5, width=1)

        if legend is True:
            handles = self.line_handles + self.scatter_handles
            labels = self.line_labels + self.scatter_labels
            # print('handles', handles, labels)
            if handles:
                self.ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        if edge is False:
            self.ax.spines['left'].set_visible(False)

    def add_annotations(self):
        if len(self.annotations) > 0:
            annotation_text = "\n".join(self.annotations)
            self.ax.annotate(annotation_text, xy=(0.15, 0.9), xycoords='axes fraction', ha='left', va='top', fontsize=9)

    def _scatter(self, df, hue='relation', legend=True, kwarg_dict=None):
        if kwarg_dict is None:
            kwarg_dict = {'s': self.scatter_points_size, 'alpha': self.scatter_points_alpha}

        self.ax = sns.scatterplot(
            data=df,
            x=self.x_var, y=self.y_var,  # 'delta_rho',
            hue=hue,
            palette=self.palette,
            ax=self.ax,
            legend=legend,
            **kwarg_dict
        )

    def update_y_extrema(self, df):
        self.min_y = df[self.y_var].min() if self.min_y is None else min(self.min_y, df[self.y_var].min())
        self.max_y = df[self.y_var].max() if self.max_y is None else max(self.max_y, df[self.y_var].max())


    def _line(self, df, hue='relation', units='surr_num',  collect_legend=True, legend=False):
        if units is not None:
            error_tuple=None
        else:
            error_tuple = ("pi", 90)
        self.ax = sns.lineplot(data=df,
                     x=self.x_var, y=self.y_var,
                     units=units,
                     hue=hue,
                     errorbar=error_tuple,
                     palette=self.palette, ax=self.ax, legend=True)

        return self.ax



class LibSizeRhoPlot(BasePlot):
    def __init__(self, y_var='rho', x_var='LibSize', units=None, lag=0, ax=None, palette=None, plot_config=None, plot_grp=None):
        # 1) Always run base init with a minimal group dict
        if isinstance(plot_config, BasePlot):
            # copy *data* attributes, not methods
            for k, v in plot_config.__dict__.items():
                setattr(self, k, v)
        else:
            base_grp = plot_grp if plot_grp is not None else {
                'y_var': y_var,
                'x_var': x_var,
                'ax': ax,
                'palette': palette
            }
            super().__init__(base_grp)

        self.lag = lag
        self.units=units

    def add_line(self, df, hue='relation', units=None,  collect_legend=True, legend=False):
        self.ax = self._line(df, hue=hue, units=units, collect_legend=collect_legend, legend=legend)
        self.update_y_extrema(df)
        self.handle_legend(collect_legend=collect_legend, legend=legend, element_type='line')
        return self.ax

    def make_classic_plot(self, outputgrp, stats_only=True, scatter=True, smoothed=False, surr_lines=False):

        if outputgrp.libsize_aggregated is None:
            print('calculating libsize rho from scratch')
            outputgrp.aggregate_libsize()
        self.palette = check_palette_syntax(self.palette, outputgrp.libsize_aggregated.full)

        outputgrp.libsize_aggregated.get_table()

        # if stats_only is False and outputgrp.delta_rho_full is None:
        #     outputgrp.calc_delta_rho(stats_out=False, full_out=True)
        #     self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_full.full)
        # elif stats_only is False:
        #     outputgrp.delta_rho_full.get_table()

        if 'relation_0' not in outputgrp.libsize_aggregated._full.schema.names:
            outputgrp.libsize_aggregated._full = add_relation_s_inferred(outputgrp.libsize_aggregated._full, relation_col='relation')

        real_lag_df = self.pull_df(outputgrp.libsize_aggregated.real, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num', 'lag', 'E', 'tau'])
        real_lag_df = real_lag_df[real_lag_df['lag'] == self.lag]
        if self.relation_scope_real is not None:
            real_lag_df = real_lag_df[real_lag_df['relation'].isin(self.relation_scope_real)]
        if smoothed := True:
            real_lag_df[self.y_var] = (
                real_lag_df.groupby('relation')[self.y_var]
                .rolling(window=5, center=True)
                .mean()
                .reset_index(level=0, drop=True)
            )
        self.add_line(real_lag_df, units='surr_num')

        surr_lag_df = self.pull_df(outputgrp.libsize_aggregated.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num', 'lag', 'E', 'tau'])
        if self.relation_scope_surr is not None:
            surr_lag_df = surr_lag_df[surr_lag_df['relation'].isin(self.relation_scope_surr)]

        for surr_var, surr_sub_df in surr_lag_df.groupby(['surr_var']):
            self.annotations.append(f'{surr_var[0]}: n={len(surr_sub_df["surr_num"].unique())}')

        self.add_line(surr_lag_df, units=self.units)

        # self.calc_top_vals(outputgrp)

        # try:
        #     if scatter is True:
        #         if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
        #             self.add_scatter(self.pull_df(outputgrp.delta_rho_full.surrogate))
        #         else:
        #             self.add_scatter(self.pull_df(outputgrp.delta_rho_stats.surrogate))
        #     if boxplot is True:
        #         if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
        #             self.add_boxplot(self.pull_df(outputgrp.delta_rho_full.surrogate))
        #         else:
        #             self.add_boxplot(self.pull_df(outputgrp.delta_rho_stats.surrogate))
        #
        #
        #             # print('made scatter plot' ,type(self.ax))
        # except Exception as e:
        #     print('no surrogate full data for scatter', e)

        outputgrp.clear_tables()

class LagPlot(BasePlot):
    """Class to create lag plots with optional scatter and highlighted points.
    Parameters
    ----------
    y_var : str, default 'delta_rho'
        The y-axis variable to plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    palette : dict or seaborn-compatible palette, optional
        Color palette for different relation categories.

    Methods
    -------
    add_scatter(df, hue='relation', legend=True)
        Adds scatter points to the plot.
    highlight_points(df, hue='relation', edgecolor="black", legend=False)
        Highlights specific points on the plot.
    add_line(df, hue='relation', units='surr_num', legend=False)
        Adds line plots to the plot.
    make_lag_plot(output, scatter=False, surr_lines=False, stats_only=True)
        Creates the lag plot with options for scatter and surrogate lines.
    Attributes
    ----------
    top_val_color : str
        Color for highlighting top values.
    bottom_val_color : str
        Color for highlighting bottom values.
    highlight_points_size : int
        Size of highlighted points.
    highlight_points_linewidth : float
        Line width of highlighted points.
    highlight_points_alpha : float
        Alpha transparency of highlighted points.
    scatter_points_size : int
        Size of scatter points.
    scatter_points_alpha : float
        Alpha transparency of scatter points.

    Examples
    --------
    >>> lag_plot = LagPlot(y_var='delta_rho', palette=my_palette)
    >>> lag_plot.make_lag_plot(output=my_output, scatter=True, surr_lines=True, stats_only=False)

    """

    def __init__(self, y_var='delta_rho', x_var='lag', ax=None, palette=None, plot_config=None, plot_grp=None):
        # 1) Always run base init with a minimal group dict
        if isinstance(plot_config, BasePlot):
            # copy *data* attributes, not methods
            for k, v in plot_config.__dict__.items():
                setattr(self, k, v)
        else:
            base_grp = plot_grp if plot_grp is not None else {
                'y_var': y_var,
                'x_var': x_var,
                'ax': ax,
                'palette': palette
            }
            super().__init__(base_grp)


        self.top_val_color = 'black'
        self.bottom_val_color = 'gray'
        self.highlight_points_size = 40
        self.highlight_points_linewidth = 1.5
        self.highlight_points_alpha = 1
        self.scatter_points_size = 20
        self.scatter_points_alpha = 0.5


    def add_boxplot(self, df, hue='relation', relation_direction='TSI', legend=False, collect_legend=True, xlims=[-20, 20]):

        kwargs = {'widths': [40/(xlims[1]-xlims[0])]*len(df.lag.unique()), 'positions':df.lag.unique()}
        self.ax = sns.boxplot(data=df[df['relation'].str.startswith(relation_direction)], x=self.x_var, y=self.y_var,
                    hue=hue, native_scale=True, linewidth=.51, ax=self.ax, dodge=True,gap=.1,
                    legend=True, palette=self.palette, whis=(5, 95), fliersize=0, **kwargs)

        self.handle_legend(collect_legend=collect_legend, legend=legend)


    def add_scatter(self, df, hue='relation', units='surr_num', collect_legend=True,legend=False, bound_quantiles=(0.05, 0.95)):

        stats_list = []
        for _, grp_df in df.groupby([self.x_var, 'relation']):
            stats_list.append(
                grp_df[(grp_df[self.y_var] > grp_df[self.y_var].quantile(bound_quantiles[0])) & (grp_df[self.y_var] < grp_df[self.y_var].quantile(bound_quantiles[1]))])

        df = pd.concat(stats_list)
        self._scatter(df, hue=hue, legend=True,
                                kwarg_dict={'s':self.scatter_points_size, 'alpha': self.scatter_points_alpha})

        self.update_y_extrema(df)

        self.handle_legend(collect_legend=collect_legend, legend=legend)

        return self.ax

    def highlight_points(self, df, hue='relation', edgecolor="black", legend=False):
        self.ax = self._scatter(df, hue=hue, legend=legend, kwarg_dict={'s': self.highlight_points_size, 'alpha': self.highlight_points_alpha, 'color': 'none',
                                                                        'edgecolor': edgecolor, 'linewidth': self.highlight_points_linewidth})
        # sns.scatterplot(ax=ax, data=top_vals,  # hue='relation',
        #                 x='lag', y=y_var, **{'s': 40, 'alpha': 1}, palette=palette, color='none', edgecolor="black",
        #                 linewidth=1.5)

    def add_line(self, df, hue='relation', units=None,  collect_legend=True, legend=False):
        self.ax = self._line(df, hue=hue, units=units, collect_legend=collect_legend, legend=legend)
        self.update_y_extrema(df)
        self.handle_legend(collect_legend=collect_legend, legend=legend, element_type='line')

        return self.ax

    def add_top_vals(self, df):
        self.highlight_points(df, hue='relation', edgecolor=self.top_val_color, legend=False)

    def add_bottom_vals(self, df):
        self.highlight_points(df, hue='relation', edgecolor=self.bottom_val_color, legend=False)

    def get_surrogate_nums(self, dset):
        gb = dset.group_by(["surr_var"]).aggregate([("surr_num", "count")])  # columns: LibSize, rho_mean
        df = gb.to_pandas()
        for _, row in df.iterrows():
            self.annotations.append(f"{row['surr_var']}: n={row['surr_num_count']}")
        # if 'surr_num' in dset.schema.names:
        #     if 'surr_var' in dset.schema.names:
        #         for surr_var, surr_var_df in df.groupby('surr_var'):
        #             self.annotations.append(f"{surr_var}: n={surr_var_df['surr_num'].nunique()}")

    def make_classic_lag_plot(self, outputgrp, stats_only=True, scatter=True, boxplot=False, surr_lines=False):
        if outputgrp.delta_rho_stats is None:
            outputgrp.calc_delta_rho(stats_out=True)
        self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_stats.full)

        if stats_only is False and outputgrp.delta_rho_full is None:
            outputgrp.calc_delta_rho(stats_out=False, full_out=True)
            self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_full.full)
        elif stats_only is False:
            outputgrp.delta_rho_full.get_table()

        if 'relation_0' not in outputgrp.delta_rho_stats._full.schema.names:
            outputgrp.delta_rho_stats._full = add_relation_s_inferred(outputgrp.delta_rho_stats._full, relation_col='relation')

        real_lag_df = self.pull_df(outputgrp.delta_rho_stats.real, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num'])
        self.add_line(real_lag_df, units=None)

        # self.calc_top_vals(outputgrp)


        try:
            if scatter is True:
                if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
                    self.add_scatter(self.pull_df(outputgrp.delta_rho_full.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num']))
                else:
                    self.add_scatter(self.pull_df(outputgrp.delta_rho_stats.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num']))
            if boxplot is True:
                if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
                    box_df = self.pull_df(outputgrp.delta_rho_full.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num'])
                else:
                    box_df = self.pull_df(outputgrp.delta_rho_stats.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num'])

                box_df['lag'] = box_df['lag'].astype(int)
                if len(box_df)>0:
                    lags = box_df['lag'].unique()
                    lags.sort()
                    if len(lags) > 1:
                        subset_lags = [lag for ik, lag in enumerate(lags) if ik % 4 == 0]
                        box_df = box_df[box_df['lag'].isin(subset_lags)]
                    self.add_boxplot(box_df)

        except Exception as e:
            print('no surrogate full data for scatter', e)

        outputgrp.clear_tables()


class SummaryGrid(GridPlot):
    def __init__(self, nrows, ncols, width_ratios=None, height_ratios=None, grid_type='heatmap'):
        super().__init__(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios, grid_type=grid_type)

        self.vlims = []
        self.cbar_ax = None
        self.cbar_label = ''
        self.marker_d = {}
        self.vlims = []
        self.palette = None
        self.sizes= (0, 400)
        # self.grid_type = 'heatmap'


    def make_colorbar(self):
        self.cbar_ax = self.get_ax(0, self.ncols - 1)

        norm = mpl.colors.Normalize(vmin=min(self.vlims), vmax=max(self.vlims))

        cbar = mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=self.palette, norm=norm)

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


class ResultsGrid(BasePlot):

    def __init__(self, relationship, sizes = (0, 400), hue_var='delta_rho',
                 y_var='tau', x_var='E', ax=None, palette=None, plot_config=None, plot_grp=None):
        # 1) Always run base init with a minimal group dict
        if isinstance(plot_config, BasePlot):
            # copy *data* attributes, not methods
            for k, v in plot_config.__dict__.items():
                setattr(self, k, v)
        else:
            base_grp = plot_grp if plot_grp is not None else {
                'y_var': y_var,
                'x_var': x_var,
                'ax': ax,
                'palette': palette
            }
            super().__init__(base_grp)


        self.relationships = relationship
        self.marker_key = {'left':self.relationships.var_x, 'right':self.relationships.var_y}
        self.marker_d = {self.marker_key['left']: MarkerStyle('o', 'left'), self.marker_key['right']: MarkerStyle('o', 'right'),
                         # 'both':MarkerStyle('o', 'full'),
                         'statistical': MarkerStyle('X'), '% deltarho <0': MarkerStyle('s', 'full'),
                         'end behavior': MarkerStyle('^', 'full')}

        self.hue_var = hue_var
        self.surr_size_var = 'perc_pos_rs_fail'
        # self.grid_df = grid_df
        self.sizes = sizes
        self.vmin = 0
        self.vmax = .4
        # self.pal = None
        self.cbar = False
        self.dyad_df = None
        self.ylabel = None
        self.xlabel = None
        self.title = None
        self.grid_type='heatmap'

    def populate_from_cellobj(self, cellobj):
        # self.dyad_df = cellobj.dyad_df.copy()
        self.ylabel = '\n'.join(cellobj.row_labels)
        self.xlabel = '\n'.join(cellobj.col_labels)
        self.title = '\n'.join(cellobj.title_labels)
        self.vmin, self.vmax = cellobj.ylims
        # print(self.vmin, self.vmax, cellobj.ylims, 'ylims')
        # print('populated from cellobj', self.ylabel, self.xlabel, self.title)

    def prep_dyad_df(self, dyad_df):
        dyad_df.sort_values([self.y_var, self.x_var], inplace=True)
        dyad_df.reset_index(inplace=True, drop=True)
        dyad_df = dyad_df[
            [self.y_var, self.x_var, self.hue_var, 'surr_ry_outperforming_frac', 'surr_rx_outperforming_frac',
             # 'perc_pos_r','perc_pos_r_top', 'deltarho_r_top','perc_pos_r_final']
             ]].copy()
        dyad_df[self.hue_var] = dyad_df.apply(lambda row: 0 if (row['surr_ry_outperforming_frac'] is None) or (row['surr_rx_outperforming_frac'] is None) else row[self.hue_var], axis=1)
        # dyad_df = dyad_df.drop_duplicates(['tau', 'E', hue_var_fill, 'TSI_p_less__maxlibsize_rho', 'temp_p_less__maxlibsize_rho'])
        dyad_df[self.hue_var].fillna(-1, inplace=True)
        if dyad_df[self.hue_var].sum() == 0:
            dyad_df = None
        return dyad_df

    def plot_heatmap(self, grid_df, ax=None):
        if self.ax is None:
            if ax is None:
                fig, self.ax = plt.subplots(figsize=(8, 6))
            else:
                self.ax = ax

        # dyad_df = self.grid_df.copy()
        dyad_df =self.prep_dyad_df(grid_df)
        if dyad_df is None:
            self.ax = None
            return self.ax

        pivot_table = dyad_df.pivot(index=self.y_var, columns=self.x_var, values=self.hue_var)
        pivot_table.fillna(-1, inplace=True)
        pivot_table.sort_index(inplace=True)

        self.ax = sns.heatmap(pivot_table, cmap=self.palette, ax=self.ax, annot=False, cbar=self.cbar, mask=pivot_table.isnull(),
                    vmin=self.vmin, vmax=self.vmax)

        return self.ax

    def add_half_moons(self, dyad_df):

        dyad_df = self.prep_dyad_df(dyad_df)
        if (dyad_df is None) or (self.ax is None):
            self.ax = None
            return

        dyad_df['E'] = dyad_df['E'] - 3.5
        dyad_df['tau'] = dyad_df['tau'] - .5
        surr_decision_gen =dyad_df.copy()

        surr_decision_gen_x = surr_decision_gen.copy()
        surr_decision_gen_x['fill_style'] = self.relationships.var_x
        surr_decision_gen_x = surr_decision_gen_x.rename(
            columns={'surr_rx_outperforming_frac': self.surr_size_var})
        # if logging is True:
        #     print(surr_decision_gen_x.sort_values(self.surr_size_var, ascending=True).head(10))

        surr_decision_gen_y = surr_decision_gen.copy()
        surr_decision_gen_y['fill_style'] = self.relationships.var_y
        surr_decision_gen_y = surr_decision_gen_y.rename(columns={'surr_ry_outperforming_frac': self.surr_size_var})
        # if logging is True:
        #     print(surr_decision_gen_y.sort_values(self.surr_size_var, ascending=True).head(10))

        surr_decision_gen = pd.concat([surr_decision_gen_x, surr_decision_gen_y])

        self.ax = sns.scatterplot(
            data=surr_decision_gen, x=self.x_var, y=self.y_var, size=self.surr_size_var, ax=self.ax,
            sizes=self.sizes, c='w',
            size_norm=(1 - .95, 1),  # this means that below values of percent_threshold, the size will be 0
            legend=True,
            style='fill_style', markers=self.marker_d,
            zorder=10,
            linewidth=1, edgecolor='w',  # this prevents the outline
        )

        self.handle_legend(collect_legend=True, legend=False, element_type='line')

    def tidy_grid(self, suptitle='', supxlabel='', supylabel=''):
        # hue_norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        # print('hue_norm', hue_norm.vmin, hue_norm.vmax)
        if self.ax is None:
            return
        self.ax.invert_yaxis()

        self.ax.set_title(self.title if self.title is not None else self.ax.get_title())# else suptitle, fontsize='large', fontweight='bold', pad=15)
        self.ax.set_xlabel(self.xlabel if self.xlabel is not None else self.ax.get_xlabel())# else supxlabel, fontsize='medium')
        self.ax.set_ylabel(self.ylabel if self.ylabel is not None else self.ax.get_ylabel())# else supylabel, fontsize='medium')
        # print('set title/xlabel/ylabel', self.title, self.xlabel, self.ylabel)
    # def make_special_legend(self):


class SimplexGrid(BasePlot):

    def __init__(self, hue_var='rho',
                 y_var='tau', x_var='E', ax=None, palette=None, plot_config=None, plot_grp=None):
        # 1) Always run base init with a minimal group dict
        if isinstance(plot_config, BasePlot):
            # copy *data* attributes, not methods
            for k, v in plot_config.__dict__.items():
                setattr(self, k, v)
        else:
            base_grp = plot_grp if plot_grp is not None else {
                'y_var': y_var,
                'x_var': x_var,
                'ax': ax,
                'palette': palette
            }
            super().__init__(base_grp)


        self.hue_var = hue_var
        self.vmin = None
        self.vmax = None
        # self.pal = None
        self.cbar = False
        self.dyad_df = None
        self.ylabel = None
        self.xlabel = None
        self.title = None
        self.cbar_ax = None
        self.cbar_label = r'$\rho$'


    def populate_from_cellobj(self, cellobj):
        # self.dyad_df = cellobj.dyad_df.copy()
        self.ylabel = '\n'.join(cellobj.row_labels)
        self.xlabel = '\n'.join(cellobj.col_labels)
        self.title = '\n'.join(cellobj.title_labels)
        self.vmin, self.vmax = cellobj.ylims
        # print(self.vmin, self.vmax, cellobj.ylims, 'ylims')
        # print('populated from cellobj', self.ylabel, self.xlabel, self.title)

    def prep_dyad_df(self, dyad_df):
        dyad_df.sort_values([self.y_var, self.x_var], inplace=True)
        dyad_df.reset_index(inplace=True, drop=True)
        dyad_df = dyad_df[
            [self.y_var, self.x_var, self.hue_var,
             # 'perc_pos_r','perc_pos_r_top', 'deltarho_r_top','perc_pos_r_final']
             ]].copy()
        # dyad_df[self.hue_var] = dyad_df.apply(lambda row: 0 if (row['surr_ry_outperforming_frac'] is None) or (row['surr_rx_outperforming_frac'] is None) else row[self.hue_var], axis=1)
        # dyad_df = dyad_df.drop_duplicates(['tau', 'E', hue_var_fill, 'TSI_p_less__maxlibsize_rho', 'temp_p_less__maxlibsize_rho'])
        dyad_df[self.hue_var].fillna(-1, inplace=True)
        return dyad_df

    def plot_heatmap(self, grid_df, ax=None):
        if self.ax is None:
            if ax is None:
                fig, self.ax = plt.subplots(figsize=(8, 6))
            else:
                self.ax = ax

        # dyad_df = self.grid_df.copy()
        dyad_df =self.prep_dyad_df(grid_df)

        pivot_table = dyad_df.pivot(index=self.y_var, columns=self.x_var, values=self.hue_var)
        # pivot_table.fillna(-1, inplace=True)
        pivot_table.sort_index(inplace=True)

        self.ax = sns.heatmap(pivot_table, cmap=self.palette, ax=self.ax, annot=False, cbar=self.cbar, mask=pivot_table.isnull(),
                    vmin=self.vmin, vmax=self.vmax)

        if (self.vmin is None) or (self.vmax is None):
            quadmesh = self.ax.collections[0]
            norm = quadmesh.norm
            if self.vmin is None:
                self.vmin = norm.vmin
            if self.vmax is None:
                self.vmax = norm.vmax

        return self.ax

    def add_half_moons(self, dyad_df):
        dyad_df = self.prep_dyad_df(dyad_df)
        dyad_df['E'] = dyad_df['E'] - 3.5
        dyad_df['tau'] = dyad_df['tau'] - .5
        surr_decision_gen =dyad_df.copy()

        surr_decision_gen_x = surr_decision_gen.copy()
        surr_decision_gen_x['fill_style'] = self.relationships.var_x
        surr_decision_gen_x = surr_decision_gen_x.rename(
            columns={'surr_rx_outperforming_frac': self.surr_size_var})
        # if logging is True:
        #     print(surr_decision_gen_x.sort_values(self.surr_size_var, ascending=True).head(10))

        surr_decision_gen_y = surr_decision_gen.copy()
        surr_decision_gen_y['fill_style'] = self.relationships.var_y
        surr_decision_gen_y = surr_decision_gen_y.rename(columns={'surr_ry_outperforming_frac': self.surr_size_var})
        # if logging is True:
        #     print(surr_decision_gen_y.sort_values(self.surr_size_var, ascending=True).head(10))

        surr_decision_gen = pd.concat([surr_decision_gen_x, surr_decision_gen_y])

        self.ax = sns.scatterplot(
            data=surr_decision_gen, x=self.x_var, y=self.y_var, size=self.surr_size_var, ax=self.ax,
            sizes=self.sizes, c='w',
            size_norm=(1 - .95, 1),  # this means that below values of percent_threshold, the size will be 0
            legend=True,
            style='fill_style', markers=self.marker_d,
            zorder=10,
            linewidth=1, edgecolor='w',  # this prevents the outline
        )

        self.handle_legend(collect_legend=True, legend=False, element_type='line')

    def tidy_grid(self, suptitle='', supxlabel='', supylabel=''):
        # hue_norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        # print('hue_norm', hue_norm.vmin, hue_norm.vmax)
        self.ax.invert_yaxis()

        self.ax.set_title(self.title if self.title is not None else self.ax.get_title())# else suptitle, fontsize='large', fontweight='bold', pad=15)
        self.ax.set_xlabel(self.xlabel if self.xlabel is not None else self.ax.get_xlabel())# else supxlabel, fontsize='medium')
        self.ax.set_ylabel(self.ylabel if self.ylabel is not None else self.ax.get_ylabel())# else supylabel, fontsize='medium')
        # print('set title/xlabel/ylabel', self.title, self.xlabel, self.ylabel)
    # def make_special_legend(self):

    def make_colorbar(self, cbar_ax=None, label=None):
        # self.cbar_ax = self.get_ax(0, self.ncols - 1)
        if cbar_ax is not None:
            self.cbar_ax = cbar_ax

        if label is not None:
            self.cbar_label = label

        if self.cbar_ax is None:
            return
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.palette),
        #                     ax=self.cbar_ax)
        cbar = mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=self.palette, norm=norm)

        # colors = cmap(np.arange(cmap.N))
        # self.cbar_ax.imshow(self.palette, extent=[0, 10, 0, 1])
        self.cbar_ax.set_ylim([self.vmin, self.vmax])
        self.cbar_ax.set_ylabel(self.cbar_label, labelpad=10)

    def tidy_grid(self, suptitle='', supxlabel='', supylabel=''):
        # hue_norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        # print('hue_norm', hue_norm.vmin, hue_norm.vmax)
        self.ax.invert_yaxis()

        self.ax.set_title(self.title if self.title is not None else self.ax.get_title())# else suptitle, fontsize='large', fontweight='bold', pad=15)
        self.ax.set_xlabel(self.xlabel if self.xlabel is not None else self.ax.get_xlabel())# else supxlabel, fontsize='medium')
        self.ax.set_ylabel(self.ylabel if self.ylabel is not None else self.ax.get_ylabel())# else supylabel, fontsize='medium')