import math
import re

import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line
# import cedarkit.utils


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

import pyarrow.compute as pc


def check_palette_syntax(palette, table, logger=None, default_color='gray'):
    if isinstance(table, pa.Table):
        schema_names = table.schema.names
    else:
        schema_names = table.collect_schema().names()

    relation_col = 'relation'
    if relation_col not in schema_names:
        relation_col = 'relation_0' if 'relation_0' in schema_names else None
    if relation_col is None:
        raise ValueError("No relation column found in table")

    if isinstance(table, pa.Table):
        relations = [r for r in pc.unique(table[relation_col]).to_pylist() if r is not None]
    else:
        relations = [r for r in table.select(relation_col).unique().collect()[relation_col].to_list() if r is not None]

    palette = dict(palette)

    def parse_relation(rel):
        rel = rel.strip()
        for word in (' influences ', ' causes ', ' reconstructs '):
            if word in rel:
                x, y = rel.split(word, 1)
                return x.strip(), word.strip(), y.strip()
        if '->' in rel:
            x, y = rel.split('->', 1)
            return x.strip(), '->', y.strip()
        return None, None, None

    for rel in relations:
        if rel in palette:
            continue

        x, kind, y = parse_relation(rel)
        if kind is None:
            palette[rel] = default_color
            if logger:
                logger.warning(f"Unrecognized relation syntax: {rel}")
            continue

        if kind in ('reconstructs', '->'):
            candidates = [
                f'{x} reconstructs {y}',
                f'{x} -> {y}',
                f'{y} influences {x}',
                f'{y} causes {x}',
            ]
        elif kind == 'influences':
            candidates = [
                f'{x} influences {y}',
                f'{x} causes {y}',
            ]
        else:
            candidates = [
                f'{x} causes {y}',
                f'{x} influences {y}',
            ]

        match = next((k for k in candidates if k in palette), None)
        palette[rel] = palette[match] if match else default_color

        if match is None and logger:
            logger.warning(f"Relation '{rel}' not found in palette keys: {list(palette.keys())}")

    return palette

# def check_palette_syntax(palette, table, logger=None, default_color='gray'):
#     relation_col = 'relation'
#     if relation_col not in table.schema.names:
#         relation_col = 'relation_0' if 'relation_0' in table.schema.names else None
#     if relation_col is None:
#         raise ValueError("No relation column found in table")
#
#     relations = [r for r in pc.unique(table[relation_col]).to_pylist() if r is not None]
#     palette = dict(palette)
#     # print(palette)
#
#     def parse_relation(rel):
#         rel = rel.strip()
#         # print(f"Parsing relation: '{rel}'")
#         for word in (' influences ', ' causes ', ' reconstructs '):
#             if word in rel:
#                 x, y = rel.split(word, 1)
#                 return x.strip(), word.strip(), y.strip()
#         if '->' in rel:
#             x, y = rel.split('->', 1)
#             return x.strip(), '->', y.strip()
#         return None, None, None
#
#     for rel in relations:
#         # print(rel)
#         if rel in palette:
#             continue
#
#         x, kind, y = parse_relation(rel)
#         # print(x, kind, y)
#         if kind is None:
#             palette[rel] = default_color
#             if logger:
#                 logger.warning(f"Unrecognized relation syntax: {rel}")
#             continue
#
#         if kind in ('reconstructs', '->'):
#             candidates = [
#                 f'{x} reconstructs {y}',
#                 f'{x} -> {y}',
#                 f'{y} influences {x}',
#                 f'{y} causes {x}',
#             ]
#         elif kind == 'influences':
#             candidates = [
#                 f'{x} influences {y}',
#                 f'{x} causes {y}',
#             ]
#         else:  # causes
#             candidates = [
#                 f'{x} causes {y}',
#                 f'{x} influences {y}',
#             ]
#
#         match = next((k for k in candidates if k in palette), None)
#         palette[rel] = palette[match] if match else default_color
#
#         if match is None and logger:
#             logger.warning(f"Relation '{rel}' not found in palette keys: {list(palette.keys())}")
#
#     return palette


# def check_palette_syntax(palette, table):
#     relation_col = 'relation'
#     if relation_col not in table.schema.names:
#         relation_col = 'relation_0' if 'relation_0' in table.schema.names else None
#     relations = pc.unique(table[relation_col]).to_pylist()
#
#     reconstructs_word = 'reconstructs' if any('->' or 'reconstructs' in r for r in relations) else None
#     rel_word = 'causes' if any('cause' or 'influence' in r for r in relations) else None
#
#     if reconstructs_word is None and rel_word is not None:
#         palette_rel_word = 'causes' if any('cause' in r for r in palette.keys()) else 'influences'
#         # new_palette = {}
#         # for k, v in palette.items():
#         #     new_key = k.replace(palette_rel_word, rel_word)
#         #     print(f"Replacing palette key '{k}' with '{new_key}'")
#         #     new_palette[new_key] = v
#         palette = {k.replace(palette_rel_word, rel_word): v for k, v in palette.items()}
#         for rel in relations:
#             if rel not in palette:
#                 palette[rel.replace(palette_rel_word, rel_word)] = 'gray'  # default color for missing keys
#                 logger.warning(f"Relation '{rel}' from data not found in palette keys: {list(palette.keys())}")
#
#     elif reconstructs_word is not None:
#         palette_recon_word = 'reconstructs' if any('reconstructs' in r for r in palette.keys()) else '->'
#         # new_palette = {}
#         # for k, v in palette.items():
#         #     new_key = k.replace(palette_recon_word, reconstructs_word)
#         #     print(f"Replacing palette key '{k}' with '{new_key}'")
#         #     new_palette[new_key] = v
#         palette = {k.replace(palette_recon_word, reconstructs_word): v for k, v in palette.items()}
#         for rel in relations:
#             if rel not in palette:
#                 palette[rel.replace(palette_recon_word, reconstructs_word)] = 'gray'  # default color for missing keys
#                 logger.warning(f"Relation '{rel}' from data not found in palette keys: {list(palette.keys())}")
#     return palette


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
    a, b = sorted(names)  # order doesn't matter for labeling; pick a stable order
    return a, b


def add_relation_s_inferred(
        table,
        x_var_name: str = None,
        y_var_name: str = None,
        surr_col: str = "surr_var",
        relation_col: str = "relation_0",
):
    if isinstance(table, pa.Table):
        # print('table schema names', table.schema.names)
        if relation_col not in table.schema.names:
            relation_col = "relation"
        if relation_col not in table.schema.names or surr_col not in table.schema.names:
            raise KeyError(f"Need columns '{relation_col}' and '{surr_col}'")

        # Prefer explicit names, then table metadata columns, then relation inference.
        if x_var_name is None and "x_var" in table.schema.names:
            x_vals = [v for v in pc.unique(table["x_var"]).to_pylist() if v is not None and str(v) != ""]
            if len(x_vals) == 1:
                x_var_name = str(x_vals[0])
        if y_var_name is None and "y_var" in table.schema.names:
            y_vals = [v for v in pc.unique(table["y_var"]).to_pylist() if v is not None and str(v) != ""]
            if len(y_vals) == 1:
                y_var_name = str(y_vals[0])

        if x_var_name is None or y_var_name is None:
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
    else:
        schema_names = table.collect_schema().names() if isinstance(table, pl.LazyFrame) else table.columns

        if relation_col not in schema_names:
            relation_col = "relation"
        if relation_col not in schema_names or surr_col not in schema_names:
            raise KeyError(f"Need columns '{relation_col}' and '{surr_col}'")

        if x_var_name is None or y_var_name is None:
            raise ValueError("x_var_name and y_var_name must be provided for the Polars branch")

        if relation_col == "relation":
            table = table.rename({"relation": "relation_0"})
            source_relation_col = "relation_0"
            target_relation_col = "relation"
        else:
            source_relation_col = relation_col
            target_relation_col = relation_col

        rel = pl.col(source_relation_col)
        surr = pl.col(surr_col)

        rel_s = (
            pl.when(surr == "neither").then(rel)
            .when(surr == "both").then(
                rel
                .str.replace_all(x_var_name, f"{x_var_name} (surr) ")
                .str.replace_all(y_var_name, f"{y_var_name} (surr) ")
            )
            .when(surr == x_var_name).then(
                rel.str.replace_all(x_var_name, f"{x_var_name} (surr) ")
            )
            .when(surr == y_var_name).then(
                rel.str.replace_all(y_var_name, f"{y_var_name} (surr) ")
            )
            .otherwise(rel)
            .str.replace_all("  ", " ")
            .str.strip_chars()
        )

        table = table.with_columns(rel_s.alias(target_relation_col))

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
