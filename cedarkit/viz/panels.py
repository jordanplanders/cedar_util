import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, Rectangle
import logging
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

logger = logging.getLogger(__name__)
try:
    from cedarkit.utils.plotting.plotting_utils import (
        check_palette_syntax,
        replace_latex_labels,
        isotope_ylabel,
        build_discrete_lag_palette,
    )
    from cedarkit.utils.cli import log_line
except ImportError:
    from utils.plotting.plotting_utils import (
        check_palette_syntax,
        replace_latex_labels,
        isotope_ylabel,
        build_discrete_lag_palette,
    )
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

        self.id_columns = ('relation', 'surr_var', 'surr_num', 'lag', 'E', 'tau')

        if grp_d is not None:
            self.populate(grp_d)

    @property
    def pull_columns(self):
        """The standard plotting frame shared by CedarKit panel plots.

        ``relation_spec`` is deliberately absent: it is an Output-level
        derived field used internally to materialize the displayed
        ``relation`` column.
        """
        return list(dict.fromkeys(
            column for column in (*self.id_columns, self.x_var, self.y_var)
            if column is not None
        ))

    def default_pull_columns(self, output_type):
        """Return the standard frame supported by an output family."""
        columns = self.pull_columns.copy()
        if output_type in {'delta_rho_full', 'delta_rho_stats'}:
            columns = [column for column in columns if column not in {'E', 'tau'}]
        return columns

    def populate(self, grp_d):
        for key, value in grp_d.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.ax is None:
            self.ax = plt.subplots(figsize=(8, 6))[1]

    def _pull_df(self, output, columns=None):
        if isinstance(output, pa.Table):
            print('output is a pa.Table')
            if columns is not None:
                output = output.select(columns)
            return output.to_pandas()

        if isinstance(output, pl.LazyFrame):
            if columns is not None:
                output = output.select(columns)
            return output.collect().to_pandas()

        if isinstance(output, pl.DataFrame):
            if columns is not None:
                output = output.select(columns)
            return output.to_pandas()

        raise TypeError(f"Unsupported output type: {type(output)}")

    def pull_df(self, outputgrp, output_type, output_scope, columns=None, relation_cats=None,
                relation_convention="pres", comparison_labels=None):
        if isinstance(outputgrp, (list, tuple)):
            if comparison_labels is not None and len(comparison_labels) != len(outputgrp):
                raise ValueError("comparison_labels must match the number of output collections.")
            frames = []
            child_columns = None if columns is None else [
                column for column in columns if column != "comparison_label"
            ]
            for index, collection in enumerate(outputgrp):
                frame = self.pull_df(
                    collection,
                    output_type,
                    output_scope,
                    columns=child_columns,
                    relation_cats=relation_cats,
                    relation_convention=relation_convention,
                )
                if frame is None:
                    raise ValueError(
                        f"Collection at index {index} has no {output_type!r} output for scope {output_scope!r}."
                    )
                if comparison_labels is not None:
                    frame["comparison_label"] = comparison_labels[index]
                frames.append(frame)
            if not frames:
                return pd.DataFrame(columns=columns)
            combined = pd.concat(frames, ignore_index=True, sort=False)
            if columns is not None:
                missing_columns = [column for column in columns if column not in combined.columns]
                if missing_columns:
                    raise ValueError(f"Combined plotting data is missing columns {missing_columns}.")
                combined = combined.loc[:, columns]
            return combined

        output_obj = None
        if output_type == 'delta_rho_stats':
            outputgrp.delta_rho_stats.get_table()
            output_obj = outputgrp.delta_rho_stats
        elif output_type == 'delta_rho_full':
            outputgrp.delta_rho_full.get_table()
            output_obj = outputgrp.delta_rho_full
        elif output_type == 'libsize_aggregated':
            outputgrp.libsize_aggregated.get_table()
            output_obj = outputgrp.libsize_aggregated
        else:
            raise ValueError(
                f"Unsupported output_type '{output_type}'. "
                "Use 'delta_rho_stats', 'delta_rho_full', or 'libsize_aggregated'."
            )

        if output_obj is None:
            return None


        self.palette = check_palette_syntax(self.palette, output_obj._full)

        if output_scope == 'real':
            output = output_obj.real
        elif output_scope == 'surrogate':
            output = output_obj.surrogate
        elif output_scope == 'full':
            output = output_obj.full
        else:
            raise ValueError(f"Unsupported output_scope '{output_scope}'. Use 'real', 'surrogate', or 'full'.")

        if output is None:
            return None

        mapped_relation_cats = []
        if relation_cats is not None:
            for relationship_id in relation_cats:
                mapped_relation_cats += outputgrp.relation_aliases(relationship_id)

            mapped_relation_cats = list(dict.fromkeys(mapped_relation_cats))

            if isinstance(output, pa.Table):
                mask = pc.is_in(output["relation"], value_set=pa.array(mapped_relation_cats))
                output = output.filter(mask)
            else:
                output = output.filter(pl.col("relation").is_in(mapped_relation_cats))

        requested_columns = list(columns) if columns is not None else self.default_pull_columns(output_type)
        source_columns = list(dict.fromkeys([
            *requested_columns,
            'relation',
            'relation_spec',
        ]))
        df = self._pull_df(output, columns=source_columns)
        outputgrp.clear_tables()

        df["relation"] = df["relation_spec"]
        relationships = getattr(outputgrp, "relationships", None)
        if relationships is not None:
            if relation_convention == "pres":
                relation_mapping = relationships.to_pres_mapping
            elif relation_convention == "calc":
                relation_mapping = relationships.to_calc_mapping
            else:
                raise ValueError(
                    f"Unsupported relation_convention '{relation_convention}'. Use 'pres' or 'calc'."
                )
            df["relation"] = df["relation"].map(lambda value: relation_mapping.get(value, value))
            if isinstance(self.palette, dict):
                for source_label, presentation_label in relation_mapping.items():
                    if source_label in self.palette and presentation_label not in self.palette:
                        self.palette[presentation_label] = self.palette[source_label]

        self.palette = check_palette_syntax(self.palette, df, logger=logger)
        if "relation_spec" not in requested_columns:
            df = df.drop(columns=["relation_spec"], errors="ignore")

        return df


    def handle_legend(self, collect_legend=True, legend=False, element_type='scatter', custom_handles=None, custom_labels=None):
        if collect_legend is True:
            if custom_handles is not None and custom_labels is not None:
                if element_type == 'line':
                    target_handles = self.line_handles
                    target_labels = self.line_labels
                else:
                    target_handles = self.scatter_handles
                    target_labels = self.scatter_labels

                for handle, label in zip(custom_handles, custom_labels):
                    if label in target_labels:
                        target_handles[target_labels.index(label)] = handle
                    else:
                        target_handles.append(handle)
                        target_labels.append(label)
            else:
                handles, labels = self.ax.get_legend_handles_labels()
                # print('handles, labels', handles, labels)
                if element_type == 'scatter':
                    for ik in range(len(handles)):
                        label=labels[-(ik+1)]
                        handle=handles[-(ik+1)]

                        if label not in self.line_labels and label not in self.scatter_labels:
                            self.scatter_handles.append(handle)
                            self.scatter_labels.append(label)

                        # if isinstance(handle, (mpl.lines.Line2D) ) is False:
                        #     if label not in self.scatter_labels:
                        #         self.scatter_handles.append(handle)
                        #         self.scatter_labels.append(label)

                elif element_type == 'line':
                    for handle, label in zip(handles, labels):
                        if label not in self.line_labels:
                            self.line_handles.append(handle)
                            self.line_labels.append(label)



        if legend is False:
            self.ax.legend().remove()

    def tidy_plot(self, legend=False, edge=True, bottom=True):
        # Axis labels
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


    def _line(self, df, hue='relation', units='surr_num',  collect_legend=True, legend=False, linewidth=2, line_alpha=1):
        if units is not None:
            error_tuple=None
        else:
            error_tuple = ("pi", 90)

        self.ax = sns.lineplot(data=df,
                     x=self.x_var, y=self.y_var,
                     units=units,
                     hue=hue,
                     errorbar=error_tuple,
                     palette=self.palette, ax=self.ax, legend=True, linewidth=linewidth, alpha=line_alpha)

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
        if collect_legend is True and units is None and hue in df.columns:
            custom_handles = []
            custom_labels = []
            for relation in pd.unique(df[hue].dropna()):
                color = self.palette.get(relation, None) if isinstance(self.palette, dict) else None
                if color is None:
                    continue
                custom_handles.append(mpl.lines.Line2D([0], [0], color=color, lw=6, alpha=0.25))
                custom_labels.append(relation)
            if custom_handles:
                self.handle_legend(
                    collect_legend=collect_legend,
                    legend=legend,
                    element_type='line',
                    custom_handles=custom_handles,
                    custom_labels=custom_labels,
                )
        return self.ax

    def make_classic_plot(self, outputgrp, stats_only=True, scatter=True, smoothed=False, surr_lines=False):

        if outputgrp.libsize_aggregated is None:
            print('calculating libsize rho from scratch')
            outputgrp.aggregate_libsize()
        self.palette = check_palette_syntax(self.palette, outputgrp.libsize_aggregated.full)

        if outputgrp.libsize_aggregated._full is None:
            outputgrp.libsize_aggregated.get_table()


        if isinstance(outputgrp.libsize_aggregated._full, pa.Table):
            schema_names = outputgrp.libsize_aggregated._full.schema.names
        else:
            schema_names = outputgrp.libsize_aggregated._full.collect_schema().names()
        has_surrogates = "surr_var" in schema_names
        if has_surrogates:
            surrogate_check = outputgrp.libsize_aggregated._full.select(
                pl.col("surr_var").ne("neither").any()
            )
            if isinstance(surrogate_check, pl.LazyFrame):
                surrogate_check = surrogate_check.collect()
            has_surrogates = surrogate_check.item()

        real_lag_df = self.pull_df(
            outputgrp,
            'libsize_aggregated',
            'real',
            relation_cats=self.relation_scope_real,
        )
        real_lag_df = real_lag_df[real_lag_df['lag'] == self.lag]
        if smoothed:
            group_cols = ["relation", "surr_var", "surr_num"]
            real_lag_df = real_lag_df.sort_values(group_cols + ["LibSize"]).copy()
            real_lag_df[self.y_var] = (
                real_lag_df.groupby(group_cols, dropna=False)[self.y_var]
                .transform(lambda values: values.rolling(window=2, center=True, min_periods=1).mean())
            )
            # real_lag_df[self.y_var] = (
            #     real_lag_df.groupby('relation')[self.y_var]
            #     .rolling(window=5, center=True)
            #     .mean()
            #     .reset_index(level=0, drop=True)
            # )
        self.add_line(real_lag_df, units='surr_num')


        if has_surrogates:
            surr_lag_df = self.pull_df(
                outputgrp,
                'libsize_aggregated',
                'surrogate',
                relation_cats=self.relation_scope_surr,
            )

            if len(surr_lag_df.lag.unique()) > 1:
                lag_d = {}
                for lag, lag_surr_df in surr_lag_df.groupby('lag'):
                    counts = [
                        surr_df.surr_num.nunique()
                        for _, surr_df in lag_surr_df.groupby('surr_var')
                    ]
                    lag_d[lag] = counts

                winner = max(
                    lag_d,
                    key=lambda lag: (
                        min(lag_d[lag]),  # biggest minimum count wins
                        sum(lag_d[lag]),  # then biggest total
                        -abs(lag),  # then closest to 0
                    )
                )
            else:
                winner = surr_lag_df['lag'].iloc[0]

            surr_lag_df = surr_lag_df[surr_lag_df['lag'] == winner].copy()

            # surr_lag_df = surr_lag_df[surr_lag_df['lag'] == self.lag]
            if smoothed:
                surr_lag_df[self.y_var] = (
                    surr_lag_df.groupby(['relation', 'surr_var', 'surr_num'])[self.y_var]
                    .rolling(window=5, center=True)
                    .mean()
                    .reset_index(level=[0, 1, 2], drop=True)
                )

            for surr_var, surr_sub_df in surr_lag_df.groupby(['surr_var']):
                annotation = f'{surr_var[0]}: n={len(surr_sub_df["surr_num"].unique())}'
                if annotation not in self.annotations:
                    self.annotations.append(annotation)

            self.add_line(surr_lag_df, units=self.units)

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


    def add_boxplot(
        self,
        df,
        hue='relation',
        relation_direction='TSI',
        legend=False,
        collect_legend=True,
        xlims=[-20, 20],
        width=1.0,
        spacing_factor=0.8,
        gap_fraction=0.08,
        box_alpha=0.45,
        line_alpha=1.0,
        linewidth=0.51,
        median_linewidth=0.75,
        outline=True,
        outline_color='black',
        outline_alpha=1.0,
        whisker_cap_fraction=0.45,
    ):

        def _half_lw_data():
            # Transform a distance of box_linewidth/2 points into data units
            # transData maps data -> display; we want the inverse for a length
            px_per_point = self.ax.get_figure().dpi / 72
            half_lw_px = (box_linewidth / 2) * px_per_point
            # One data unit in pixels (x direction)
            data_to_px = self.ax.transData.transform([1, 0]) - self.ax.transData.transform([0, 0])
            px_per_data_unit = data_to_px[0]
            return half_lw_px / px_per_data_unit

        if self.ax is None:
            self.ax = plt.subplots(figsize=(8, 6))[1]

        preserve_xlim = self.ax.has_data()
        old_xlim = self.ax.get_xlim()
        plot_df = df[df['relation'].str.startswith(relation_direction)].copy()
        cols = [self.x_var, self.y_var] + ([hue] if hue else [])
        plot_df = plot_df[cols].dropna()

        box_linewidth = linewidth if outline else 0
        hlw = _half_lw_data()

        if plot_df.empty:
            self.handle_legend(collect_legend=collect_legend, legend=legend)
            return self.ax

        x_values = np.array(sorted(plot_df[self.x_var].unique()), dtype=float)
        if len(x_values) == 0:
            self.handle_legend(collect_legend=collect_legend, legend=legend)
            return self.ax

        if len(x_values) == 1:
            group_width = width
        else:
            min_spacing = np.min(np.diff(x_values))
            # Hard cap: group can't exceed available slot
            # spacing_factor provides breathing room between adjacent groups
            max_group_width = spacing_factor * min_spacing
            group_width = min(width, max_group_width)

        if hue is None:
            hue_levels = [None]
        else:
            hue_levels = list(pd.unique(plot_df[hue]))

        def _get_color(group):
            if isinstance(self.palette, dict):
                return self.palette.get(group, plt.get_cmap("tab10")(0))
            cmap = plt.get_cmap("tab10")
            return cmap(hue_levels.index(group) % 10)

        legend_handles = []
        legend_labels = []

        for x in x_values:
            sub = plot_df[plot_df[self.x_var] == x]
            if hue is None:
                grouped = [(None, sub)]
            else:
                grouped = [
                    (group, sub[sub[hue] == group])
                    for group in hue_levels
                    if not sub[sub[hue] == group].empty
                ]

            n_boxes = len(grouped)
            if n_boxes == 0:
                continue

            # Compute box_width first, then derive the total group footprint
            gap = group_width * gap_fraction if n_boxes > 1 else 0
            box_width = (group_width - gap * (n_boxes - 1)) / n_boxes
            actual_group_width = n_boxes * box_width + (n_boxes - 1) * gap  # == group_width, but explicit
            left0 = x - actual_group_width / 2

            for j, (group, gdf) in enumerate(grouped):
                y = gdf[self.y_var].to_numpy(dtype=float)
                y = y[np.isfinite(y)]
                if len(y) == 0:
                    continue

                box_center = left0 + j * (box_width + gap) + box_width / 2
                box_low, box_high = np.percentile(y, (25, 75))
                whisker_low, whisker_high = np.percentile(y, (5, 95))
                color = _get_color(group)
                facecolor = mpl.colors.to_rgba(color, alpha=box_alpha)
                if outline_color == 'native':
                    edge_base = color
                else:
                    edge_base = outline_color
                edgecolor = mpl.colors.to_rgba(edge_base, alpha=outline_alpha) if outline else 'none'


                rect = Rectangle(
                    (box_center - box_width / 2 + hlw, box_low),
                    box_width - 2 * hlw,
                    box_high - box_low,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=box_linewidth,
                    zorder=3,
                )

                rect.set_clip_path(rect)
                self.ax.add_patch(rect)

                if whisker_low < box_low:
                    self.ax.vlines(
                        box_center,
                        whisker_low,
                        box_low,
                        color='black',
                        linewidth=linewidth,
                        alpha=line_alpha,
                        zorder=3,
                    )

                if whisker_high > box_high:
                    self.ax.vlines(
                        box_center,
                        box_high,
                        whisker_high,
                        color='black',
                        linewidth=linewidth,
                        alpha=line_alpha,
                        zorder=3,
                    )

                cap_width = box_width * whisker_cap_fraction
                self.ax.hlines(
                    [whisker_low, whisker_high],
                    box_center - cap_width / 2,
                    box_center + cap_width / 2,
                    color='black',
                    linewidth=linewidth,
                    alpha=line_alpha,
                    zorder=3,
                )

                median = np.median(y)
                self.ax.hlines(
                    median,
                    box_center - box_width / 2,
                    box_center + box_width / 2,
                    color='black',
                    linewidth=median_linewidth,
                    alpha=line_alpha,
                    zorder=4,
                )

                if group not in legend_labels:
                    legend_handles.append(mpl.lines.Line2D([0], [0], color=color, lw=6, alpha=box_alpha))
                    legend_labels.append(group)

        self.ax.autoscale(axis="y")
        if preserve_xlim:
            self.ax.set_xlim(old_xlim)
        else:
            self.ax.set_xlim(x_values.min() - group_width, x_values.max() + group_width)

        self.handle_legend(
            collect_legend=collect_legend,
            legend=legend,
            element_type='line',
            custom_handles=legend_handles,
            custom_labels=legend_labels,
        )
        return self.ax


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

    def add_line(self, df, hue='relation', units=None,  collect_legend=True, legend=False, linewidth=2, line_alpha=1):
        self.ax = self._line(df, hue=hue, units=units, collect_legend=collect_legend, legend=legend, linewidth=linewidth, line_alpha=line_alpha)
        self.update_y_extrema(df)
        self.handle_legend(collect_legend=collect_legend, legend=legend, element_type='line')

        return self.ax

    def add_top_vals(self, df):
        self.highlight_points(df, hue='relation', edgecolor=self.top_val_color, legend=False)

    def add_bottom_vals(self, df):
        self.highlight_points(df, hue='relation', edgecolor=self.bottom_val_color, legend=False)

    def get_surrogate_nums(self, dset):
        if isinstance(dset, pa.Table):
            gb = dset.group_by(["surr_var"]).aggregate([("surr_num", "count")])
            df = gb.to_pandas()
        elif isinstance(dset, pl.LazyFrame):
            df = dset.group_by("surr_var").agg(pl.col("surr_num").count().alias("surr_num_count")).collect().to_pandas()
        elif isinstance(dset, pl.DataFrame):
            df = dset.group_by("surr_var").agg(pl.col("surr_num").count().alias("surr_num_count")).to_pandas()
        else:
            raise TypeError(f"Unsupported dataset type: {type(dset)}")

        for _, row in df.iterrows():
            self.annotations.append(f"{row['surr_var']}: n={row['surr_num_count']}")
        # if 'surr_num' in dset.schema.names:
        #     if 'surr_var' in dset.schema.names:
        #         for surr_var, surr_var_df in df.groupby('surr_var'):
        #             self.annotations.append(f"{surr_var}: n={surr_var_df['surr_num'].nunique()}")

    def make_classic_lag_plot(self, outputgrp, stats_only=True, scatter=True, boxplot=False, surr_lines=False,
                              relation_scope=None, comparison_labels=None, hue='relation'):

        if isinstance(outputgrp, (list, tuple)):
            real_lag_df = self.pull_df(
                outputgrp,
                'delta_rho_stats',
                'real',
                relation_cats=relation_scope,
                comparison_labels=comparison_labels,
            )
            if not real_lag_df.empty:
                if hue == 'comparison_label':
                    comparison_values = set(real_lag_df['comparison_label'].dropna())
                    if not isinstance(self.palette, dict) or not comparison_values.issubset(self.palette):
                        self.palette = None
                self.add_line(real_lag_df, hue=hue, units=None)

            if scatter or boxplot:
                use_full = (
                    stats_only is False
                    and all(getattr(collection, 'delta_rho_full', None) is not None for collection in outputgrp)
                )
                surrogate_output_type = 'delta_rho_full' if use_full else 'delta_rho_stats'
                surrogate_df = self.pull_df(
                    outputgrp,
                    surrogate_output_type,
                    'surrogate',
                    relation_cats=relation_scope,
                    comparison_labels=comparison_labels,
                )
                if hue == 'comparison_label' and not surrogate_df.empty:
                    comparison_values = set(surrogate_df['comparison_label'].dropna())
                    if not isinstance(self.palette, dict) or not comparison_values.issubset(self.palette):
                        self.palette = None
                if scatter and not surrogate_df.empty:
                    self.add_scatter(surrogate_df, hue=hue)
                if boxplot and not surrogate_df.empty:
                    box_df = surrogate_df.copy()
                    box_df['lag'] = box_df['lag'].astype(int)
                    lags = np.sort(box_df['lag'].unique())
                    if len(lags) > 1:
                        box_df = box_df[box_df['lag'].isin(lags[::4])]
                    self.add_boxplot(box_df, hue=hue)
            return self.ax

        # if outputgrp.delta_rho_stats is None:
        #     outputgrp.calc_delta_rho(stats_out=True)
        outputgrp.delta_rho_stats.get_table()
        self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_stats.full)

        if stats_only is False and outputgrp.delta_rho_full is None:
            # outputgrp.calc_delta_rho(stats_out=False, full_out=True)
            outputgrp.delta_rho_full.get_table()

            self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_full.full)

        elif stats_only is False:
            outputgrp.delta_rho_full.get_table()

        real_lag_df = self.pull_df(
            outputgrp,
            'delta_rho_stats',
            'real',
            relation_cats=relation_scope,
        )
        self.add_line(real_lag_df, units=None)


        try:
            if scatter is True:
                if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
                    self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_full.surrogate)
                    self.add_scatter(self.pull_df(
                        outputgrp,
                        'delta_rho_full',
                        'surrogate',
                        relation_cats=relation_scope,
                    ))
                else:
                    self.palette = check_palette_syntax(self.palette, outputgrp.delta_rho_stats.surrogate)
                    self.add_scatter(self.pull_df(
                        outputgrp,
                        'delta_rho_stats',
                        'surrogate',
                        relation_cats=relation_scope,
                    ))
            if boxplot is True:
                if stats_only is False and outputgrp.delta_rho_full is not None and len(outputgrp.delta_rho_full.surrogate) > 0:
                    box_df = self.pull_df(
                        outputgrp,
                        'delta_rho_full',
                        'surrogate',
                        relation_cats=self.relation_scope_surr,
                    )
                else:
                    log_line(logger, 'boxplot with stats', indent=0,
                             log_type="debug")
                    box_df = self.pull_df(
                        outputgrp,
                        'delta_rho_stats',
                        'surrogate',
                        relation_cats=self.relation_scope_surr,
                    )
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
            print(e)

        outputgrp.clear_tables()


class ResultsGrid(BasePlot):

    def __init__(self, relationship, sizes = (0, 400), hue_var='delta_rho',
                 y_var='tau', x_var='E', ax=None, palette=None, plot_config=None, plot_grp=None,norm=None,
                 lag_mode='unrestricted', x_cutoff=0, show_corner=False, show_half_moons=True,outline_color=None,
                 show_peak_circles=False, peak_window_halfwidth=3, lag_filter=None):
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
        self.norm = norm
        self.dyad_df = None
        self.lag_mode = lag_mode
        self.x_cutoff = x_cutoff
        self.show_corner = show_corner
        self.show_half_moons = show_half_moons
        self.show_peak_circles = show_peak_circles
        self.peak_window_halfwidth = peak_window_halfwidth
        self.corner_size = 0.4
        self.corner_column = 'selected_lag_pos'
        self.sharpness_column = 'peak_sharpness'
        self.circle_sizes = (30, 380)
        self.tie_marker = True
        self.discrete_lag_info = None
        self._x_lookup = {}
        self._y_lookup = {}
        self._plotted_df = None
        self._plotted_norm = None
        self.use_discrete_lag = False
        self.outline_color = outline_color
        # self.ylabel = None
        # self.xlabel = None
        # self.title = None
        self.grid_type='heatmap'
        self.lag_filter = lag_filter

    def populate_from_cellobj(self, cellobj):
        # self.dyad_df = cellobj.dyad_df.copy()
        self.ylabel = '\n'.join(cellobj.row_labels)
        self.xlabel = '\n'.join(cellobj.col_labels)
        self.title = '\n'.join(cellobj.title_labels)
        self.vmin, self.vmax = cellobj.ylims
        # print(self.vmin, self.vmax, cellobj.ylims, 'ylims')
        # print('populated from cellobj', self.ylabel, self.xlabel, self.title)

    def prep_dyad_df(self, dyad_df):
        # Legacy behavior used by existing ResultGrid notebooks.
        dyad_df.sort_values([self.y_var, self.x_var], inplace=True)
        dyad_df.reset_index(inplace=True, drop=True)

        # if self.lag_filter is not None:
        if callable(self.lag_filter) and 'peak_end' in dyad_df.columns:
            dyad_df['peak_end'] = dyad_df['peak_end'].astype(float)
            dyad_df.loc[dyad_df['peak_end'].apply(self.lag_filter), self.hue_var] = np.nan
            dyad_df[self.hue_var] = dyad_df.apply(lambda row: 0 if (row['surr_ry_outperforming_frac'] is None) or (row['surr_rx_outperforming_frac'] is None) else row[self.hue_var], axis=1)

        dyad_df = dyad_df[
            [self.y_var, self.x_var, self.hue_var, 'surr_ry_outperforming_frac', 'surr_rx_outperforming_frac',
             # 'perc_pos_r','perc_pos_r_top', 'deltarho_r_top','perc_pos_r_final']
             ]].copy()

        if self.show_half_moons is True:
            dyad_df[self.hue_var] = dyad_df.apply(lambda row: 0 if (row['surr_ry_outperforming_frac'] is None) or (row['surr_rx_outperforming_frac'] is None) else row[self.hue_var], axis=1)
        # dyad_df = dyad_df.drop_duplicates(['tau', 'E', hue_var_fill, 'TSI_p_less__maxlibsize_rho', 'temp_p_less__maxlibsize_rho'])

        if dyad_df[self.hue_var].isna().any():
            print(dyad_df[dyad_df[self.hue_var].isna()], 'dyad_df hue_var with nans')
        # dyad_df[self.hue_var].fillna(-1, inplace=True)

        if dyad_df[self.hue_var].sum() == 0:
            dyad_df[self.hue_var] = np.nan#None
        return dyad_df

    def prep_optimal_lag_df(self, dyad_df):
        required_cols = [self.y_var, self.x_var, 'selected_lag']
        if dyad_df is None or any(col not in dyad_df.columns for col in required_cols):
            return None
        tmp = dyad_df.copy()
        tmp.sort_values([self.y_var, self.x_var], inplace=True)
        tmp.reset_index(inplace=True, drop=True)
        if self.corner_column not in tmp.columns:
            tmp[self.corner_column] = np.nan
        if self.sharpness_column not in tmp.columns:
            tmp[self.sharpness_column] = np.nan
        if 'has_tie' not in tmp.columns:
            tmp['has_tie'] = False
        if 'surr_ry_outperforming_frac' not in tmp.columns:
            tmp['surr_ry_outperforming_frac'] = None
        if 'surr_rx_outperforming_frac' not in tmp.columns:
            tmp['surr_rx_outperforming_frac'] = None
        return tmp

    def _build_axis_lookup(self, pivot_table):
        self._x_lookup = {x_val: ix for ix, x_val in enumerate(pivot_table.columns.tolist())}
        self._y_lookup = {y_val: iy for iy, y_val in enumerate(pivot_table.index.tolist())}

    def plot_heatmap(self, grid_df, ax=None):
        if self.ax is None:
            if ax is None:
                fig, self.ax = plt.subplots(figsize=(8, 6))
            else:
                self.ax = ax

        # self.use_discrete_lag = self._is_optimal_lag_mode(grid_df)
        # self._plotted_norm = None
        # if self.use_discrete_lag:
        #     dyad_df = self.prep_optimal_lag_df(grid_df)
        #     if dyad_df is None:
        #         self.ax = None
        #         return self.ax
        #     lag_vals = dyad_df['selected_lag'].dropna().astype(int).tolist()
        #     if self.show_corner and self.corner_column in dyad_df.columns:
        #         lag_vals.extend(dyad_df[self.corner_column].dropna().astype(int).tolist())
        #     if self.discrete_lag_info is None:
        #         self.discrete_lag_info = build_discrete_lag_palette(lag_vals, palette=self.palette)
        #
        #     pivot_lag = dyad_df.pivot(index=self.y_var, columns=self.x_var, values='selected_lag')
        #     pivot_lag.sort_index(inplace=True)
        #     self._build_axis_lookup(pivot_lag)
        #
        #     if self.show_peak_circles:
        #         base = pd.DataFrame(
        #             np.zeros_like(pivot_lag.values, dtype=float),
        #             index=pivot_lag.index,
        #             columns=pivot_lag.columns,
        #         )
        #         self.ax = sns.heatmap(
        #             base,
        #             cmap=mpl.colors.ListedColormap(['white']),
        #             ax=self.ax,
        #             annot=False,
        #             cbar=False,
        #             vmin=0,
        #             vmax=1,
        #             linewidths=0.25,
        #             linecolor='0.9',
        #         )
        #         if len(self.ax.collections) > 0:
        #             self._plotted_norm = self.ax.collections[0].norm
        #     else:
        #         pivot_idx = self._to_lag_index_df(pivot_lag, self.discrete_lag_info['lag_to_index'])
        #         print('heatmap', self.discrete_lag_info['lag_to_index'])
        #         self.ax = sns.heatmap(
        #             pivot_idx,
        #             cmap=self.discrete_lag_info['cmap'],
        #             ax=self.ax,
        #             annot=False,
        #             cbar=self.cbar,
        #             mask=pivot_idx.isnull(),
        #             vmin=-0.5,
        #             vmax=len(self.discrete_lag_info['lags']) - 0.5,
        #             linecolor = self.outline_color if self.outline_color is not None else 'none',
        #         )
        #         if len(self.ax.collections) > 0:
        #             self._plotted_norm = self.ax.collections[0].norm

            # if self.show_corner and self.lag_mode == 'compare':
            #     self.add_corner_overlay(dyad_df)
            # if self.show_peak_circles:
            #     self.add_peak_circles(dyad_df)
            # if self.tie_marker:
            #     self.add_tie_markers(dyad_df)

            # self._plotted_df = dyad_df
            # return self.ax

        dyad_df = self.prep_dyad_df(grid_df)
        if dyad_df is None:
            self.ax = None
            return self.ax

        pivot_table = dyad_df.pivot(index=self.y_var, columns=self.x_var, values=self.hue_var)
        # pivot_table.fillna(-1, inplace=True)
        pivot_table.sort_index(inplace=True)
        self._build_axis_lookup(pivot_table)
        heatmap_kwargs = dict(
            cmap=self.palette,
            ax=self.ax,
            annot=False,
            cbar=self.cbar,
            mask=pivot_table.isnull(),
            linecolor=self.outline_color if self.outline_color is not None else 'none',
            linewidths=0.25 if self.outline_color is not None else 0,
        )
        if self.norm is not None:
            heatmap_kwargs['norm'] = self.norm
        else:
            heatmap_kwargs['vmin'] = self.vmin
            heatmap_kwargs['vmax'] = self.vmax

        # self.ax.set_facecolor("lightgray")
        self.ax = sns.heatmap(pivot_table, **heatmap_kwargs)

        if len(self.ax.collections) > 0:
            self._plotted_norm = self.ax.collections[0].norm
        self._plotted_df = dyad_df

        return self.ax

    def add_half_moons(self, dyad_df):
        if self.show_half_moons is False:
            return
        if isinstance(dyad_df, pd.DataFrame) and 'selected_lag' in dyad_df.columns:
            dyad_df = self.prep_optimal_lag_df(dyad_df)
        else:
            dyad_df = self.prep_dyad_df(dyad_df)
        if (dyad_df is None) or (self.ax is None):
            self.ax = None
            return

        dyad_df = dyad_df.copy()
        if len(self._x_lookup) > 0 and len(self._y_lookup) > 0:
            dyad_df[self.x_var] = dyad_df[self.x_var].map(lambda v: self._x_lookup.get(v, np.nan) + 0.5 if v in self._x_lookup else np.nan)
            dyad_df[self.y_var] = dyad_df[self.y_var].map(lambda v: self._y_lookup.get(v, np.nan) + 0.5 if v in self._y_lookup else np.nan)
            dyad_df = dyad_df[dyad_df[self.x_var].notna() & dyad_df[self.y_var].notna()].copy()
        else:
            dyad_df[self.x_var] = dyad_df[self.x_var] - 3.5
            dyad_df[self.y_var] = dyad_df[self.y_var] - 0.5
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
            linewidth=.7,
            edgecolor=self.outline_color if self.outline_color is not None else 'none',
            rasterized=True # for publication
            # linewidth=0.25 if self.outline_color is not None else 0,
        )

        self.handle_legend(collect_legend=True, legend=False, element_type='line')

    def mask_nan_cells(self, grid_df):
        if self.ax is None:
            return
        if isinstance(grid_df, pd.DataFrame) and 'selected_lag' in grid_df.columns:
            dyad_df = self.prep_optimal_lag_df(grid_df)
        else:
            dyad_df = self.prep_dyad_df(grid_df)
        if (dyad_df is None) or (self.ax is None):
            self.ax = None
            return
        pivot_table = dyad_df.pivot(index=self.y_var, columns=self.x_var, values=self.hue_var)
        pivot_table.sort_index(inplace=True)
        mask = pivot_table.isnull()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask.iloc[i, j]:
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='whitesmoke', edgecolor='none', zorder=1000)
                    self.ax.add_patch(rect)


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


class SimplexGrid(BasePlot):

    def __init__(self, hue_var='rho',
                 y_var='tau', x_var='E', ax=None, palette=None, plot_config=None, plot_grp=None, norm=None):
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
        self.norm = norm
        self.dyad_df = None
        self._plotted_norm = None
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

        heatmap_kwargs = dict(
            cmap=self.palette,
            ax=self.ax,
            annot=False,
            cbar=self.cbar,
            mask=pivot_table.isnull(),
        )
        if self.norm is not None:
            heatmap_kwargs['norm'] = self.norm
        else:
            heatmap_kwargs['vmin'] = self.vmin
            heatmap_kwargs['vmax'] = self.vmax

        self.ax = sns.heatmap(pivot_table, **heatmap_kwargs)
        if len(self.ax.collections) > 0:
            self._plotted_norm = self.ax.collections[0].norm
        else:
            self._plotted_norm = None

        if ((self.vmin is None) or (self.vmax is None)) and self._plotted_norm is not None:
            norm = self._plotted_norm
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

    def make_colorbar(self, cbar_ax=None, label=None):
        # self.cbar_ax = self.get_ax(0, self.ncols - 1)
        if cbar_ax is not None:
            self.cbar_ax = cbar_ax

        if label is not None:
            self.cbar_label = label

        if self.cbar_ax is None:
            return
        if self._plotted_norm is not None:
            norm = self._plotted_norm
        elif self.norm is not None:
            norm = self.norm
        else:
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.palette),
        #                     ax=self.cbar_ax)
        cbar = mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=self.palette, norm=norm)

        # colors = cmap(np.arange(cmap.N))
        # self.cbar_ax.imshow(self.palette, extent=[0, 10, 0, 1])
        norm_vmin = getattr(norm, 'vmin', None)
        norm_vmax = getattr(norm, 'vmax', None)
        if norm_vmin is not None and norm_vmax is not None:
            self.cbar_ax.set_ylim([norm_vmin, norm_vmax])
            self.vmin = norm_vmin
            self.vmax = norm_vmax
        else:
            self.cbar_ax.set_ylim([self.vmin, self.vmax])
        self.cbar_ax.set_ylabel(self.cbar_label, labelpad=10)
