import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import compute as pc
import pyarrow as pa
import re
import matplotlib as mpl

def check_palette_syntax(palette, table):
    relation_col = 'relation'
    if relation_col not in table.schema.names:
        relation_col = 'relation_0' if 'relation_0' in table.schema.names else None
    relations = pc.unique(table[relation_col]).to_pylist()
    rel_word = 'causes' if any('cause' in r for r in relations) else 'influences'
    palette_rel_word = 'causes' if any('cause' in r for r in palette.keys()) else 'influences'
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

# dictionary mapping location (row, col) to key
# dictionary mapping key to Output object

#wrap grid?
# subfigure rows

# E x tau grid: {(E, tau): Object}, Location grid {(row, col): (E, tau)}
# E_rows = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7}
# tau_cols = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5}
#
class GridCell:
    def __init__(self, row, col, output=None):
        self.row = row
        self.col = col
        self.occupied = False
        self.row_labels=[]
        self.col_labels=[]
        self.cell_labels=[]
        self.output = output
        self.annotations = []
        self.y_lims = []
        annotations = []


class GridPlot:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.title = None
        self.occupied_dict = {}
        self.ax_grid = {}
        self.gridspec_kw = None#{'wspace': 0.07, 'hspace': 0.07} #gridspec_kw={'width_ratios': [2, 1]}
        self.scatter_handles = []
        self.scatter_labels = []
        self.line_handles = []
        self.line_labels = []
        self.fig = None
        self.subfigs = []
        self.ylims = []
        self.xlims = []
        # self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # self.fig.tight_layout(pad=3.0)

    def make_grid(self, fig=None, figsize=None):

        self.fig = fig if fig is not None else plt.figure(
            figsize=figsize if figsize is not None else (5 * self.ncols, 4 * self.nrows))
        self.subfigs = self.fig.subfigures(self.nrows, 1, wspace=0.07, hspace=0.07) if self.nrows > 1 else [self.fig]

        for row in range(self.nrows):
            subfig = self.subfigs[row] if self.nrows > 1 else self.subfigs[0]
            axes = subfig.subplots(1, self.ncols) if self.ncols > 1 else [subfig.add_subplot(1, 1, 1)]
            for col in range(self.ncols):
                self.ax_grid[(row, col)] = axes[col] if self.ncols > 1 else axes[0]
                self.occupied_dict[(row, col)] = False

    def get_ax(self, row, col):
        return self.ax_grid.get((row, col), None)

    def set_ax(self, row, col, ax):
        self.ax_grid[(row, col)] = ax
        self.occupied_dict[(row, col)] = True

    def get_ax_row(self, row):
        return [self.ax_grid.get((row, col), None) for col in range(self.ncols)]

    def get_subfig(self, row):
        return self.subfigs[row] if self.nrows > 1 else self.subfigs[0]

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


    def tidy_rows(self):
        # self.remove_empty()
        ylims = (min(self.ylims), max(self.ylims)) if self.ylims else (None, None)
        for ik, subfig in enumerate(self.subfigs):
            for ip, ax in enumerate(subfig.axes):
                ax.grid(False)
                ax.tick_params(axis='y', length=5, width=1)
                ax.tick_params(axis='x', length=5, width=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                if ip > 0:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_ylabel('')

                ax.set_ylim(ylims)
                if ik < len(self.subfigs) - 1:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.spines['bottom'].set_visible(False)

                if ik >0:
                    ax.set_title('')

                if len(self.xlims) == 2:
                    ax.set_xlim(self.xlims)

            yticks = subfig.axes[0].get_yticks()
            subfig.axes[0].spines['left'].set_bounds(yticks[1], yticks[-2])
            ylabel = subfig.axes[0].get_ylabel()
            ylabel_parts = ylabel.split('\n')
            if len(ylabel_parts) > 1:
                supylabel = ylabel_parts[0]
                ylabel = '\n'.join(ylabel_parts[1:])
                subfig.supylabel(supylabel, x=0.05, y=0.55, fontsize='large', fontweight='bold')
                subfig.axes[0].set_ylabel(ylabel)

            # self.subfigs[ik]= subfig
            # subfig.axes[0].yaxis.set_ticks([])
            # print('set yticks to none')
                # ax.spines['left'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)

        plt.tight_layout()

    def add_legend(self, bbox_to_anchor=(1.05, 1), loc='upper left'):
        handles = self.line_handles + self.scatter_handles
        labels = self.line_labels + self.scatter_labels
        ax_legend = self.subfigs[0].axes[-1] if self.nrows > 1 else self.subfigs[0].axes[0]
        if handles:
            ax_legend.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc=loc)

    def remove_empty(self):
        for (row, col), occupied in self.occupied_dict.items():
            if not occupied:
                ax = self.get_ax(row, col)
                if ax is not None:
                    ax.remove()#('off')

        self.subfigs = [subfig for subfig in self.subfigs if len(subfig.axes) > 0]
        # for ik, subfig in enumerate(self.subfigs):
        #     if len(subfig.axes)==0:
        #         subfig.remove()

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

class LagPlot:
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


    def __init__(self, y_var='delta_rho', ax=None, palette=None):
        self.y_var = y_var
        self.ax = ax if ax is not None else plt.subplots(figsize=(8, 6))[1]
        self.palette = palette
        self.top_val_color = 'black'
        self.bottom_val_color = 'gray'
        self.highlight_points_size = 40
        self.highlight_points_linewidth = 1.5
        self.highlight_points_alpha = 1
        self.scatter_points_size = 20
        self.scatter_points_alpha = 0.5

        self.scatter_handles = []
        self.scatter_labels = []

        self.line_handles = []
        self.line_labels = []
        self.min_y = None
        self.max_y = None

        self.annotations = []

    def add_boxplot(self, df, hue='relation', legend=False, collect_legend=True):
        sns.boxplot(data=df[df['relation'].str.startswith('TSI')], x="lag", y=self.y_var,
                    hue=hue, native_scale=True, linewidth=.51, ax=self.ax,
                    legend=True, width=2, palette=self.palette, whis=(5, 95), fliersize=0)

        if collect_legend is True:
            handles, labels = self.ax.get_legend_handles_labels()
            # print('handles, labels', handles, labels)
            for ik in range(len(handles)):
                label=labels[-(ik+1)]
                handle=handles[-(ik+1)]
            # for handle, label in zip(handles, labels):
                if isinstance(handle, mpl.lines.Line2D) is False:
                    if label not in self.scatter_labels:
                        self.scatter_handles.append(handle)
                        self.scatter_labels.append(label)

        if legend is False:
            self.ax.legend().remove()

    def _scatter(self, df, hue='relation', legend=True, kwarg_dict=None):
        if kwarg_dict is None:
            kwarg_dict = {'s': self.scatter_points_size, 'alpha': self.scatter_points_alpha}

        self.ax = sns.scatterplot(
            data=df,
            x='lag', y=self.y_var,  # 'delta_rho',
            hue=hue,
            palette=self.palette,
            ax=self.ax,
            legend=legend,
            **kwarg_dict
        )

    def add_scatter(self, df, hue='relation', collect_legend=True,legend=False, bound_quantiles=(0.05, 0.95)):
        # print('got to add scatter', len(df))
        stats_list = []
        for _, grp_df in df.groupby(['lag', 'relation']):
            stats_list.append(
                grp_df[(grp_df[self.y_var] > grp_df[self.y_var].quantile(bound_quantiles[0])) & (grp_df[self.y_var] < grp_df[self.y_var].quantile(bound_quantiles[1]))])

        df = pd.concat(stats_list)
        self._scatter(df, hue=hue, legend=True,
                                kwarg_dict={'s':self.scatter_points_size, 'alpha': self.scatter_points_alpha})

        self.update_y_extrema(df)
        # .scatterplot(
        #     data=df,
        #     x='lag', y=self.y_var,  # 'delta_rho',
        #     hue=hue,
        #     palette=self.palette,
        #     ax=self.ax,
        #     legend=legend,
        #     **{'s': 20, 'alpha': .5}
        # ))
        # return self.ax
        if collect_legend is True:
            handles, labels = self.ax.get_legend_handles_labels()
            for ik in range(len(handles)):
                label=labels[-(ik+1)]
                handle=handles[-(ik+1)]
            # for handle, label in zip(handles, labels):
                if label not in self.scatter_labels:
                    self.scatter_handles.append(handle)
                    self.scatter_labels.append(label)

        if legend is False:
            self.ax.legend().remove()

    def highlight_points(self, df, hue='relation', edgecolor="black", legend=False):
        self.ax = self._scatter(df, hue=hue, legend=legend, kwarg_dict={'s': self.highlight_points_size, 'alpha': self.highlight_points_alpha, 'color': 'none',
                                                                        'edgecolor': edgecolor, 'linewidth': self.highlight_points_linewidth})
        # sns.scatterplot(ax=ax, data=top_vals,  # hue='relation',
        #                 x='lag', y=y_var, **{'s': 40, 'alpha': 1}, palette=palette, color='none', edgecolor="black",
        #                 linewidth=1.5)

    def add_line(self, df, hue='relation', units='surr_num',  collect_legend=True, legend=False):
        self.ax = sns.lineplot(data=df,
                     x='lag', y=self.y_var,
                     units=units,
                     hue=hue,
                     palette=self.palette, ax=self.ax, legend=True)

        self.update_y_extrema(df)

        if collect_legend is True:
            handles, labels = self.ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if handle not in self.scatter_handles:
                    self.line_handles.append(handle)
                    self.line_labels.append(label)
        if legend is False:
            self.ax.legend().remove()
        # return self.ax


    def update_y_extrema(self, df):
        self.min_y = df[self.y_var].min() if self.min_y is None else min(self.min_y, df[self.y_var].min())
        self.max_y = df[self.y_var].max() if self.max_y is None else max(self.max_y, df[self.y_var].max())


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

    def make_classic_lag_plot(self, output, stats_only=True, scatter=True, boxplot=False, surr_lines=False):
        if output.delta_rho_stats is None:
            output.calc_delta_rho(stats_out=True)
        self.palette = check_palette_syntax(self.palette, output.delta_rho_stats.full)
        # self.get_surrogate_nums(output.delta_rho_stats.surrogate)
        if stats_only is False and output.delta_rho_full is None:
            output.calc_delta_rho(stats_out=False, full_out=True)
            self.palette = check_palette_syntax(self.palette, output.delta_rho_full.full)
        elif stats_only is False:
            output.delta_rho_full.get_table()
        # print(output.delta_rho_stats._full.schema.names)

        if 'relation_0' not in output.delta_rho_stats._full.schema.names:
            output.delta_rho_stats._full = add_relation_s_inferred(output.delta_rho_stats._full, relation_col='relation')
        self.add_line(output.delta_rho_stats.real.select(['lag', self.y_var, 'surr_var', 'surr_num', 'relation']).to_pandas(), units=None)
        # print('made line plot')
        try:
            if scatter is True:
                if stats_only is False and output.delta_rho_full is not None and len(output.delta_rho_full.surrogate) > 0:
                    self.add_scatter(output.delta_rho_full.surrogate.select(['lag', self.y_var, 'relation', 'surr_var', 'surr_num']).to_pandas())
                else:
                    self.add_scatter(output.delta_rho_stats.surrogate.select(['lag', self.y_var, 'relation', 'surr_var', 'surr_num']).to_pandas())
            if boxplot is True:
                if stats_only is False and output.delta_rho_full is not None and len(output.delta_rho_full.surrogate) > 0:
                    self.add_boxplot(output.delta_rho_full.surrogate.select(['lag', self.y_var, 'relation', 'surr_var', 'surr_num']).to_pandas())
                else:
                    self.add_boxplot(output.delta_rho_stats.surrogate.select(['lag', self.y_var, 'relation', 'surr_var', 'surr_num']).to_pandas())

                    # print('made scatter plot' ,type(self.ax))
        except Exception as e:
            print('no surrogate full data for scatter', e)

        output.clear_tables()


    def tidy_plot(self, legend=False, edge=True, bottom=True):
        # Axis labels
        self.ax.set_xlabel('lag')
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

        if len(self.annotations) > 0:
            annotation_text = "\n".join(self.annotations)
            self.ax.annotate(annotation_text, xy=(0.15, 0.9), xycoords='axes fraction', ha='left', va='top', fontsize=9)
        # if bottom is False:
        #     self.ax.spines['bottom'].set_visible(False)


        # if handles:
        #     # ax.legend(handles, labels, **({} if scatter else {'title': None}))
        #     return self.ax

    # def make_lag_plot_full(ax, stats_df, palette, y_var='delta_rho', scatter=False):
    #     """
    #     Draws lag-vs-Δρ (delta_rho) for each relation category using seaborn.
    #     Parameters
    #     ----------
    #     ax : matplotlib.axes.Axes
    #     stats_df : pandas.DataFrame
    #         Must contain columns: 'lag', 'delta_rho', and 'relation'.
    #     palette : dict or seaborn-compatible palette
    #     scatter : bool, default False
    #         If True, overlay scatter points on the line.
    #     """
    #     # Ensure required columns exist
    #
    #     if not {'lag', 'relation'}.issubset(stats_df.columns):
    #         if y_var not in stats_df.stat.unique():
    #             raise ValueError(f"stats_df must contain 'lag', '{y_var}', and 'relation' columns")
    #
    #     stats_df = stats_df[stats_df['stat'] == y_var]
    #     stats_df = stats_df.rename(columns={'rho': y_var})
    #
    #     if scatter:
    #         stats_list = []
    #         for _, grp_df in stats_df.groupby(['lag', 'relation']):
    #             stats_list.append(grp_df[(grp_df[y_var] > grp_df[y_var].quantile(0.05)) & (
    #                         grp_df[y_var] < grp_df[y_var].quantile(0.95))])
    #         _stats_df = pd.concat(stats_list)
    #         sns.scatterplot(
    #             data=_stats_df,
    #             x='lag', y=y_var,  # 'delta_rho',
    #             hue='relation',
    #             palette=palette,
    #             ax=ax,
    #             legend=True,
    #             **{'s': 20, 'alpha': .5}
    #         )
    #
    #     sns.lineplot(
    #         data=stats_df[~stats_df['relation'].str.contains('surr')],
    #         x='lag', y=y_var,  # 'maxlibsize_rho',
    #         hue='relation',
    #         palette=palette,
    #         legend=False,
    #         ax=ax
    #     )
    #
    #     # Axis labels
    #     ax.set_xlabel('lag')
    #     ax.set_ylabel(y_var.replace('delta_', 'Δ').replace('rho', 'ρ'))
    #
    #     available_ylabel = ax.get_ylabel()
    #     available_ylabel = available_ylabel.replace('rho', "ρ").replace('delta', "Δ")
    #     ax.set_ylabel(available_ylabel)
    #     return ax

