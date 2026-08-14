import hashlib
import json
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line

KEY_COLS = [
    "E","tau","Tp","lag",#"relation","forcing","responding",
    "pset_id","surr_var","surr_num",
    "x_id","x_age_model_ind","x_var",
    "y_id","y_age_model_ind","y_var",
]


def check_existence_in_table(parquet_df, trait_d):
    if 'x_id' in parquet_df.columns:
        parquet_df['col_var_id'] = parquet_df['x_id']
    if 'y_id' in parquet_df.columns:
        parquet_df['target_var_id']= parquet_df['y_id']

    parquet_df = parquet_df[[col for col in parquet_df.columns if col in trait_d.keys()]]
    trait_d = {key: value for key, value in trait_d.items() if key in parquet_df.columns}
    mask = pd.Series([True] * len(parquet_df))
    for k, v in trait_d.items():
        mask &= (parquet_df[k] == v)

    exists = mask.any()
    return exists


def combine_column(table, name):
    return table.column(name).combine_chunks()


def groupify_array(arr):
    # Input: Pyarrow/Numpy array
    # Output:
    #   - 1. Unique values
    #   - 2. Count per unique
    #   - 3. Sort index
    #   - 4. Begin index per unique
    dic, counts = np.unique(arr, return_counts=True)
    sort_idx = np.argsort(arr)
    return dic, counts, sort_idx, [0] + np.cumsum(counts)[:-1].tolist()



def columns_to_array(table, columns):
    columns = ([columns] if isinstance(columns, str) else list(set(columns)))
    if len(columns) == 1:
        #return combine_column(table, columns[0]).to_numpy(zero_copy_only=False)
        return (combine_column(table, columns[0]).to_numpy(zero_copy_only=False))
    else:
        values = [c.to_numpy() for c in table.select(columns).itercolumns()]
        return np.array(list(map(hash, zip(*values))))


def drop_duplicates(table, on=[], keep='first'):
    # Gather columns to arr
    arr = columns_to_array(table, (on if on else table.column_names))

    # Groupify
    dic, counts, sort_idxs, bgn_idxs = groupify_array(arr)

    # Gather idxs
    if keep == 'last':
        idxs = (np.array(bgn_idxs) - 1)[1:].tolist() + [len(sort_idxs) - 1]
    elif keep == 'first':
        idxs = bgn_idxs
    elif keep == 'drop':
        idxs = [i for i, c in zip(bgn_idxs, counts) if c == 1]
    return table.take(sort_idxs[idxs])


def make_uid(row: pd.Series) -> str:
    blob = json.dumps({k: (None if pd.isna(row[k]) else row[k]) for k in KEY_COLS},
                      sort_keys=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


as_len1_array = lambda x: as_lenN_array(x, 1)


def as_lenN_array(x, n):

    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return pa.array([None] * n, type=pa.float64())
    if isinstance(x, (int, np.integer)):
        return pa.array([int(x)] * n, type=pa.int64())
    if isinstance(x, (float, np.floating)):
        return pa.array([float(x)] * n, type=pa.float64())

    if isinstance(x, pa.Array):
        if len(x) == n:
            return x
        elif len(x) == 1:
            return pa.repeat(x, n)
        else:
            raise ValueError("Cannot broadcast array of length {} to length {}".format(len(x), n))
    # For other types (e.g., str), repeat the value n times
    return pa.array([x] * n)
