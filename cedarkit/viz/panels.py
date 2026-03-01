import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import logging

logger = logging.getLogger(__name__)
try:
    from cedarkit.utils.plotting.plotting_utils import check_palette_syntax, add_relation_s_inferred, replace_latex_labels, isotope_ylabel
    from cedarkit.utils.cli import log_line
except ImportError:
    from utils.plotting.plotting_utils import check_palette_syntax, add_relation_s_inferred, replace_latex_labels, isotope_ylabel
    from utils.cli.logging import log_line


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
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.y_var = None #y_var
        self.x_var = None # x_var
        self.palette = None #palette
        self.ylabel = None
        self.xlabel = None
        self.title = None
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
        self.ax.set_ylabel(self.y_var.replace('rho_', 'ρ'))

        available_ylabel = self.ax.get_ylabel()
        available_ylabel = available_ylabel.replace('_', ' ')
        self.ylabel = replace_latex_labels(available_ylabel)
        self.ax.set_ylabel(self.ylabel)

        xlabel_available = self.ax.get_xlabel()
        xlabel_available = xlabel_available.replace('_', ' ')
        self.xlabel = replace_latex_labels(xlabel_available)
        self.ax.set_xlabel(self.xlabel)

        title_available = self.ax.get_title()
        title_available = title_available.replace('_', ' ')
        self.title = replace_latex_labels(title_available)
        self.ax.set_title(self.title)

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
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.top_val_color = 'black'
        self.bottom_val_color = 'gray'
        self.highlight_points_size = 40
        self.highlight_points_linewidth = 1.5
        self.highlight_points_alpha = 1
        self.scatter_points_size = 20
        self.scatter_points_alpha = 0.5


    def add_boxplot(self, df, hue='relation', relation_direction='TSI', legend=False, collect_legend=True, xlims=[-20, 20]):

        kwargs = {}#{'widths': [40/(xlims[1]-xlims[0])]*len(df.lag.unique()), 'positions':df.lag.unique()}
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
        outputgrp.delta_rho_stats.get_table()
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
                    log_line(logger, 'boxplot with stats', indent=0,
                             log_type="debug")
                    box_df = self.pull_df(outputgrp.delta_rho_stats.surrogate, columns = [self.x_var, self.y_var, 'relation', 'surr_var', 'surr_num'])
                    log_line(logger, f'box_df size before lag filtering: {len(box_df)}', indent=0, log_type="debug")

                box_df['lag'] = box_df['lag'].astype(int)
                if len(box_df)>0:
                    lags = box_df['lag'].unique()
                    lags.sort()
                    if len(lags) > 1:
                        subset_lags = [lag for ik, lag in enumerate(lags) if ik % 4 == 0]
                        box_df = box_df[box_df['lag'].isin(subset_lags)]
                    self.add_boxplot(box_df)
                    log_line(logger, f'box_df lags used: {box_df["relation"].unique()}', indent=0, log_type="debug")
                    log_line(logger, [f'box_df size after lag filtering: {len(box_df)}', box_df.head()], indent=0, log_type="debug")
            # print('made scatter plot' ,type(self.ax))
        except Exception as e:
            print('no surrogate full data for scatter', e)

        outputgrp.clear_tables()


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

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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
        # self.ylabel = None
        # self.xlabel = None
        # self.title = None
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

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.hue_var = hue_var
        self.vmin = None
        self.vmax = None
        # self.pal = None
        self.cbar = False
        self.dyad_df = None
        # self.ylabel = None
        # self.xlabel = None
        # self.title = None
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

