import math
import re

import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
import polars as pl
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
    from cedarkit.utils.workflow.process_output import relation_candidates
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line
    from utils.workflow.process_output import relation_candidates


def font_resizer(context='paper', multiplier=1.0, rc=None):
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


def check_palette_syntax(palette, table, logger=None, default_color='gray'):
    """Align palette keys with relation strings found in plotting tables.

    The plotting layer may encounter both calc-facing and presentation-facing relation
    strings. Arrow notation is treated as an operation/reconstruction spelling, not as
    a forward causal spelling.
    """
    if isinstance(table, pd.DataFrame):
        schema_names = table.columns
    else:
        schema_names = table.collect_schema().names()

    relation_col = 'relation'
    if relation_col not in schema_names:
        relation_col = 'relation_0' if 'relation_0' in schema_names else None
    if relation_col is None:
        raise ValueError("No relation column found in table")

    if isinstance(table, pd.DataFrame):
        relations = [r for r in table[relation_col].unique() if r is not None]
    else:
        relation_values = table.select(relation_col).unique()

        if isinstance(relation_values, pl.LazyFrame):
            relation_values = relation_values.collect()

        relations = [r for r in relation_values[relation_col].to_list() if r is not None]

        # relations = [r for r in table.select(relation_col).unique().collect()[relation_col].to_list() if r is not None]
    palette = {} if palette is None else dict(palette)

    for rel in relations:
        if rel in palette:
            continue

        candidates = relation_candidates(rel)
        if candidates is None:
            palette[rel] = default_color
            if logger:
                logger.warning(f"Unrecognized relation syntax: {rel}")
            continue

        match = next((k for k in candidates if k in palette), None)
        palette[rel] = palette[match] if match else default_color

        if match is None and logger:
            logger.warning(f"Relation '{rel}' not found in palette keys: {list(palette.keys())}")

    return palette


def build_discrete_lag_palette(lags, palette='coolwarm'):
    """Build a stable discrete lag palette and normalization for integer lag bins."""
    lag_values = sorted({int(lag) for lag in lags if lag is not None and np.isfinite(lag)})
    if len(lag_values) == 0:
        lag_values = [0]

    if isinstance(palette, mpl.colors.Colormap):
        base_cmap = palette
    elif isinstance(palette, (list, tuple)) and len(palette) > 0:
        if len(palette) >= len(lag_values):
            colors = list(palette[:len(lag_values)])
            cmap = mpl.colors.ListedColormap(colors)
            n = len(lag_values)
            boundaries = np.arange(-0.5, n + 0.5, 1)
            norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
            lag_to_index = {lag: idx for idx, lag in enumerate(lag_values)}
            index_to_lag = {idx: lag for lag, idx in lag_to_index.items()}
            return {
                'lags': lag_values,
                'lag_to_index': lag_to_index,
                'index_to_lag': index_to_lag,
                'cmap': cmap,
                'norm': norm,
            }
        base_cmap = mpl.cm.get_cmap('coolwarm')
    else:
        base_cmap = mpl.cm.get_cmap(palette if isinstance(palette, str) else 'coolwarm')

    # Sign-aware diverging mapping:
    # - negative lags interpolate from min_lag -> 0 over [0.0, 0.5]
    # - positive lags interpolate from 0 -> max_lag over [0.5, 1.0]
    # - zero lag maps to center 0.5
    min_lag = min(lag_values)
    max_lag = max(lag_values)
    colors = []
    for lag in lag_values:
        if lag == 0:
            pos = 0.5
        elif lag < 0:
            if min_lag < 0:
                # Linear from min_lag -> 0 maps to 0.0 -> 0.5.
                pos = 0.5 * ((lag - min_lag) / (0 - min_lag))
            else:
                # Defensive fallback when no negative domain exists.
                pos = 0.5
        else:
            if max_lag > 0:
                # Linear from 0 -> max_lag maps to 0.5 -> 1.0.
                pos = 0.5 + 0.5 * (lag / max_lag)
            else:
                # Defensive fallback when no positive domain exists.
                pos = 0.5
        colors.append(base_cmap(float(np.clip(pos, 0.0, 1.0))))

    cmap = mpl.colors.ListedColormap(colors)
    n = len(lag_values)
    boundaries = np.arange(-0.5, n + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    lag_to_index = {lag: idx for idx, lag in enumerate(lag_values)}
    index_to_lag = {idx: lag for lag, idx in lag_to_index.items()}
    return {
        'lags': lag_values,
        'lag_to_index': lag_to_index,
        'index_to_lag': index_to_lag,
        'cmap': cmap,
        'norm': norm,
    }


# def int_yticks_within_ylim(ymin, ymax):
#     # Find all integer values within the current limits
#     ticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
#     # Ensure at least 2 ticks (for degenerate ranges)
#     if len(ticks) < 2:
#         ticks = np.array([np.floor(ymin), np.ceil(ymax)])
#     return ticks.astype(int)


# def replace_supylabel(label):
#     label = label.replace('Doering', 'Döring')
#     return label


# def int_yticks_from_ylim(ymin, ymax):
#     # Ensure ymin < ymax
#     if ymin == ymax:
#         ymin -= 0.5
#         ymax += 0.5
#
#     # Compute rough range and ideal tick spacing
#     yrange = ymax - ymin
#     rough_spacing = yrange / 2  # aim for ~3 ticks total (2 intervals)
#
#     # Round spacing to nearest "nice" integer (1, 2, 5, 10, etc.)
#     exp = math.floor(math.log10(rough_spacing))
#     base = rough_spacing / (10 ** exp)
#     if base < 1.5:
#         nice_base = 1
#     elif base < 3.5:
#         nice_base = 1
#     elif base < 7.5:
#         nice_base = 5
#     else:
#         nice_base = 10
#     spacing = nice_base * (10 ** exp)
#
#     # Compute tick positions
#     tick_start = math.floor(ymin / spacing) * spacing
#     tick_end = math.ceil(ymax / spacing) * spacing
#     ticks = np.arange(tick_start-spacing, tick_end + spacing, spacing)
#
#     # Ensure at least 2 ticks
#     if len(ticks) < 2:
#         ticks = np.array([math.floor(ymin), math.ceil(ymax)])
#     elif len(ticks) == 2:
#         # Try to add a middle tick if possible
#         mid = np.mean(ticks)
#         if mid.is_integer():
#             ticks = np.array([ticks[0], mid, ticks[1]])
#
#     return ticks.astype(int)


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


def replace_latex_labels(label):
    if not isinstance(label, str):
        return label

    text = label

    # Normalize malformed math wrappers before token replacement.
    text = re.sub(r'\${2,}', '$', text)

    # Keep already-math-mode chunks unchanged by replacing only bare-word tokens.
    token_rules = {
        'delta': r'$\\delta$',
        'Delta': r'$\\Delta$',
        'tau': r'$\\tau$',
        'rho': r'$\\rho$',
    }
    for token, repl in token_rules.items():
        pattern = rf'(?<![A-Za-z\\]){token}(?![A-Za-z])'
        text = re.sub(pattern, repl, text)

    # Specific scientific aliases.
    text = re.sub(r'(?<![A-Za-z\\])d18O(?![A-Za-z])', r'$\\delta^{18}O$', text)
    text = re.sub(r'(?<![A-Za-z])Wm2(?![A-Za-z])', r'W/m$^{2}$', text)

    # Exponent helpers outside existing math commands.
    text = re.sub(r'(?<![\$\\])\^18', r'$^{18}$', text)
    text = re.sub(r'(?<![\$\\])\^2', r'$^{2}$', text)

    return text
