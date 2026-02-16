import math
import re

import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyarrow as pa
import pyarrow.compute as pc
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
    if '$\delta' not in label:
        label = label.replace('delta', r'$\delta$')
    # label = label.replace('excess', r'$d$-excess')
    # label = label.replace('D', r'$D$')
    # label = label.replace('O', r'$O$')
    label = label.replace('Wm2',r'W/m$^{2}$')
    label = label.replace('d18O',r'$\delta^{18}O$')
    if '$^18' not in label:
        label = label.replace('^18', r'$^{18}$')
    if '$^2' not in label:
        label = label.replace('^2', r'$^{2}$')
    if '$\Delta' not in label:
        label = label.replace('Delta', r'$\Delta$')
    if '$\tau' not in label:
        label = label.replace('tau', r'$\tau$')
    if '$\rho' not in label:
        label = label.replace('rho', r'$\rho$')
    return label