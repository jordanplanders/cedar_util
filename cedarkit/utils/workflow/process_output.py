import pandas as pd
import logging
import re
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line


_RELATION_PATTERNS = (
    ("operation", "->", r"^\s*(.*?)\s*->\s*(.*?)\s*$"),
    ("operation", "→", r"^\s*(.*?)\s*→\s*(.*?)\s*$"),
    ("operation", "=>", r"^\s*(.*?)\s*=>\s*(.*?)\s*$"),
    ("operation", "reconstructs", r"^\s*(.*?)\s+reconstructs\s+(.*?)\s*$"),
    ("causal", "causes", r"^\s*(.*?)\s+causes\s+(.*?)\s*$"),
    ("causal", "influences", r"^\s*(.*?)\s+influences\s+(.*?)\s*$"),
)


def parse_relation(relation: str) -> dict | None:
    """Decode one persisted CedarKit relationship string."""
    if relation is None:
        return None
    text = str(relation).strip()
    for relation_type, token, pattern in _RELATION_PATTERNS:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            if lhs and rhs:
                return {
                    "lhs": lhs,
                    "rhs": rhs,
                    "relation_type": relation_type,
                    "token": token,
                }
    return None


def infer_relation_variables(relations) -> tuple[str, str]:
    """Infer CedarKit's two participant variable names from relation strings."""
    names = set()
    for relation in relations:
        parsed = parse_relation(relation)
        if parsed is not None:
            names.update((parsed["lhs"], parsed["rhs"]))
    if len(names) != 2:
        raise ValueError(f"Could not infer exactly two variable names from relations; found: {sorted(names)}")
    return tuple(sorted(names))


def relation_candidates(relation: str) -> list[str] | None:
    """Return equivalent calc, causal, and arrow spellings for palette lookup."""
    parsed = parse_relation(relation)
    if parsed is None:
        return None
    lhs, rhs = parsed["lhs"], parsed["rhs"]
    relation_type, token = parsed["relation_type"], parsed["token"]
    if relation_type == "operation":
        candidates = [
            f"{lhs} {token} {rhs}" if token in {"->", "→", "=>"} else f"{lhs} reconstructs {rhs}",
            f"{lhs} reconstructs {rhs}", f"{lhs} -> {rhs}", f"{lhs} => {rhs}",
            f"{rhs} influences {lhs}", f"{rhs} causes {lhs}",
        ]
    else:
        candidates = [
            f"{lhs} {token} {rhs}" if token in {"causes", "influences"} else f"{lhs} causes {rhs}",
            f"{lhs} causes {rhs}", f"{lhs} influences {rhs}",
            f"{rhs} reconstructs {lhs}", f"{rhs} -> {lhs}", f"{rhs} => {lhs}",
        ]
    return list(dict.fromkeys(candidates))

def unpack_ccm_output(CrossMapList_num):
    translate_d = {'columns': 'forcing', 'target': 'responding'}
    df_subs = []
    dta = CrossMapList_num['predictStats']

    for lib_size in dta.keys():
        df_sub = pd.DataFrame(dta[lib_size])
        df_sub['LibSize'] = lib_size  # Add the LibSize as a column
        df_subs.append(df_sub)
    df_sub = pd.concat(df_subs)

    # if isinstance(CrossMapList_num['columns'], list):
    #     forcing = ' '.join(CrossMapList_num['columns'])
    # else:
    #     forcing = CrossMapList_num['columns']
    responding = ' '.join(CrossMapList_num['columns']).strip('')
    forcing = ' '.join(CrossMapList_num['target']).strip('')

    df_sub['forcing'] =  forcing
    df_sub['responding'] = responding

    df_sub['relation'] = f'{forcing} causes {responding}'

    return df_sub

def add_meta_data(ccm_out, _ccm_out_df, train_ind_i, train_ind_f, lag=0, add_cols=None):
    _ccm_out_df['ind_i'] = train_ind_i
    _ccm_out_df['ind_f'] = train_ind_f
    _ccm_out_df['E'] = ccm_out.E
    _ccm_out_df['tau'] = ccm_out.tau
    _ccm_out_df['Tp'] = ccm_out.Tp
    _ccm_out_df['lag'] = lag
    if add_cols is not None:
        if not isinstance(add_cols, list):
            add_cols = [add_cols]
        for key in add_cols:
            if key == 'target':
                value = ' '.join(ccm_out.__dict__['target']).strip('')
            elif key == 'columns':
                value = ' '.join(ccm_out.__dict__['columns']).strip('')
            else:
                value =ccm_out.__dict__[key]
            _ccm_out_df[key] = value

    return _ccm_out_df
