import os, re, json, hashlib, sys
from collections import defaultdict
from pathlib import Path
from itertools import product
import time
from functools import reduce

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc

import gc

# from data_objects import extract_from_pattern
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.location_helpers import *
from utils.data_access import *

from data_obj.data_objects import DataGroup
# from tmp_utils.path_utils import set_calc_path, set_output_path, template_replacement

KEY_COLS = [
    "E","tau","Tp","lag",#"relation","forcing","responding",
    "pset_id","surr_var","surr_num",
    "x_id","x_age_model_ind","x_var",
    "y_id","y_age_model_ind","y_var",
]

# def check_csv(output_file_name):
#     if '.csv' not in output_file_name:
#         output_file_name = f'{output_file_name}.csv'
#     return output_file_name

def _parse_surr(token: str, x_var: str, y_var: str):
    """
    Parses suffix token to determine surrogate variable and index.
    Parameters:
        token (str): Suffix token indicating surrogate type and index. (e.g. 'neither0', 'TSI147', 'temp033', 'both12').
        x_var (str): Name of the first variable (e.g., 'temp').
        y_var (str): Name of the second variable (e.g., 'TSI').

    Returns:
        (surr_var, surr_num) where surr_var ∈ {'neither', x_var, y_var, 'both'}

    Used by:
        package_calc_grp_results_to_parquet
    """

    lab_l = token.lower()
    xv, yv = x_var.lower(), y_var.lower()

    if "neither" in lab_l:
        return "neither", 0
    if "both" in lab_l:
        return "both", 99999 # use a large number to indicate both but without a specific index
    if xv in lab_l:
        num = lab_l.replace(xv, '').strip()
        if num.isdigit():
            num = int(num)
        return x_var, num
    if yv in lab_l:
        num = lab_l.replace(yv, '').strip()
        if num.isdigit():
            num = int(num)
        return y_var, num
    print(f"Warning: Unrecognized suffix token '{token}', treating as 'neither0'")


def _make_uid(row: pd.Series) -> str:
    blob = json.dumps({k: (None if pd.isna(row[k]) else row[k]) for k in KEY_COLS},
                      sort_keys=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()

# def check_existance_in_table(table, trait_d):
#     if table is None:
#         return False
#     if table.num_rows == 0:
#         return False
#     try:
#         mask_list = [pc.equal(table[key], value) for key, value in trait_d.items() if key in table.schema.names]
#         if mask_list:
#             mask = reduce(pc.and_, mask_list)
#             filtered_table = table.filter(mask)
#         else:
#             filtered_table = table
#     except:
#         print('failed to filter table with', trait_d, file=sys.stderr, flush=True)
#         return False

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

def extract_from_pattern(filename: str, pattern_str: str):
    """
    Extracts parameter values from filename based on a pattern string.
    Example:
        extract_from_pattern("E4_tau1_lag-5.parquet", "E{E}_tau{tau}_lag{lag}")
        -> {'E': 4, 'tau': 1, 'lag': -5}
    """
    # Convert format specifiers like {E}, {tau}, {lag} into named regex groups
    # regex = re.sub(r"\{(\w+)\}", lambda m: f"(?P<{m.group(1)}>-?\\d+)", pattern_str)
    regex = re.sub(
        r"\{(\w+)\}",
        lambda m: f"(?P<{m.group(1)}>[-+]?\d+(?:\.\d+)?|[A-Za-z_][\\w-]*)",
        pattern_str
    )

    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match pattern '{pattern_str}'")

    # Convert all extracted values to integers
    parts_d = {k: v for k, v in match.groupdict().items()}
    parts_d = {k: int(v) if v.lstrip('-').isdigit() else v for k, v in parts_d.items()}
    return parts_d

# drop duplicates in parquet table
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

f = np.vectorize(hash)
def columns_to_array(table, columns):
    columns = ([columns] if isinstance(columns, str) else list(set(columns)))
    if len(columns) == 1:
        #return combine_column(table, columns[0]).to_numpy(zero_copy_only=False)
        return f(combine_column(table, columns[0]).to_numpy(zero_copy_only=False))
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


# def get_col_var_and_target_var(config, parts_d):
#     col_var = config.get_dynamic_attr("{var}.var", parts_d['col_var_id'])
#     target_var = config.get_dynamic_attr("{var}.var", parts_d['target_var_id'])
#     return col_var, target_var


def setup_conversion_from_calc_grp(output_dir, config, calc_grp_d):
    '''
    Setup paths and variables for conversion from CSV to Parquet for a given calculation group.
    Parameters:
        output_dir (Path): Base output directory.
        config (dict): Configuration dictionary.
        parts_d (dict): Calculation group dictionary.

    Returns:
        dict: Dictionary containing paths and variables needed for conversion.

    Used by:
        package_calc_grp_results_to_parquet
    '''
    parts_d = calc_grp_d.copy()
    # construct path pattern
    fallback_E_tau_grp_pattern = 'knn_{knn}/tp_{Tp}/{col_var_id}_{target_var_id}/E{E}_tau{tau}'
    E_tau_grp_pattern = config.output.parquet.dir_structure if config is not None else fallback_E_tau_grp_pattern

    # directory of existing CSV input
    e_tau_dir_read = output_dir/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)

    # update parts_d as determined from file structure with values from config (in case they differ)
    parts_d['col_var_id']=config.col.var_id
    parts_d['target_var_id']=config.target.var_id
    # directory of future parquet output
    e_tau_dir_write = output_dir/'parquet'/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)

    # pull from config
    col_var = config.col.var
    target_var = config.target.var

    return {'e_tau_dir_read': e_tau_dir_read, 'e_tau_dir_write':e_tau_dir_write, 'parts_d': parts_d, 'col_var': col_var, 'target_var': target_var, 'config': config}


def package_calc_grp_results_to_parquet(
    e_tau_dir_read: Path,
    e_tau_dir_write: Path,
    parts_d: str,
    col_var: str,            # e.g., 'temp'
    target_var: str,         # e.g., 'TSI'
    config: dict,
    x_age_model_ind: int | None = None,  # integer pointers from YAML
    y_age_model_ind: int | None = None,

):
    '''
    Packages calculation group results from CSV files into Parquet format (E-tau-lag units).
    Parameters:
        e_tau_dir_read (Path): Directory path to read CSV files from.
        e_tau_dir_write (Path): Directory path to write Parquet files to.
        parts_d (dict): Calculation group dictionary containing parameters like E, tau, Tp, knn, etc.
        col_var (str): Name of the column variable (e.g., 'temp').
        target_var (str): Name of the target variable (e.g., 'TSI').
        config (dict): Configuration dictionary.
        x_age_model_ind (int, optional): Age model index for column variable. Defaults to None.
        y_age_model_ind (int, optional): Age model index for target variable. Defaults to None.

    Returns:
        (write_paths, existing): Tuple containing list of written Parquet file paths and list of existing file paths.

    Used by:
        Main script block.
    '''
    # currently parts_d is expected to be the content of a calc_grp, but is parsed and reformed into grp_d rather than as a pass through

    E = parts_d.get('E', None)
    tau = parts_d.get('tau', None)
    Tp = parts_d.get('Tp', 1)
    knn = parts_d.get('knn', None)
    lag = parts_d.get('lag', None)
    col_var_id = config.col.var_id#parts_d.get('col_var_id', None)
    target_var_id = config.target.var_id#parts_d.get('target_var_id', None)

    existing = []
    write_paths = []

    print(e_tau_dir_read, e_tau_dir_read.exists(), e_tau_dir_read.is_dir(), file=sys.stdout, flush=True)
    # if CSV input directory does not exist, return
    if not e_tau_dir_read.exists() or not e_tau_dir_read.is_dir():
        print(f"Directory {e_tau_dir_read} does not exist or is not a directory", file=sys.stderr, flush=True)
        print(f"Directory {e_tau_dir_read} does not exist or is not a directory", file=sys.stdout, flush=True)
        return write_paths, existing

    lag_dir = None
    lag_dir_d = defaultdict(list)
    if lag is not None:
        if (e_tau_dir_read / f'lag_{lag}').exists() is True:
            lag_dir = e_tau_dir_read / f'lag_{lag}'
        elif (e_tau_dir_read / f'lag{lag}').exists() is True:
            lag_dir = e_tau_dir_read / f'lag{lag}'

        if lag_dir is not None:
            lag_dir_d[lag] = [lag_dir / fn for fn in os.listdir(lag_dir) if fn.endswith('.csv')]
    else:
        lag_dirs = [entry for entry in sorted(os.listdir(e_tau_dir_read)) if entry.startswith("lag")]
        if len(lag_dirs) == 0:
            print(f"No lag* subdirectories found under {e_tau_dir_read}", file=sys.stdout, flush=True)
            return write_paths, existing
        else:
            print(f"Found {len(lag_dirs)} lag* subdirectories under {e_tau_dir_read}",file=sys.stdout, flush=True)
            e_tau_dir_write.mkdir(exist_ok=True, parents=True)

        # gather CSV files under each lag directory
        for entry in lag_dirs:
            lag_dir = Path(os.path.join(e_tau_dir_read, entry))
            if os.path.isdir(lag_dir) is True:
                lag = int(entry.replace('lag_','').replace('lag',''))
                lag_dir_d[lag]+= [lag_dir/fn for fn in os.listdir(lag_dir) if fn.endswith('.csv')]

    # process each lag directory, gathering records checking to see if they have already been added to the target parquet file, finally writing to Parquet
    for lag, csvs in lag_dir_d.items():
        records = []
        sub_existing =[]

        grp_d = parts_d.copy()
        grp_d.update({'lag': lag})

        file_name_parquet = template_replace(config.output.parquet.file_format, grp_d, return_replaced=False)
        write_path = e_tau_dir_write/ f"{file_name_parquet}.parquet"
        write_path_file_valid = os.path.exists(write_path)

        # check existing parquet to see what has been recorded
        if write_path_file_valid is True:
            existing_parquet_table = ds.dataset(str(write_path), format="parquet").to_table()
            print('\texisting_parquet_table rows:', existing_parquet_table.num_rows,existing_parquet_table.schema.names, file=sys.stdout, flush=True)
            recorded_parquet = drop_duplicates(existing_parquet_table, on=['E', 'tau', 'lag', 'Tp', 'knn', 'surr_var', 'surr_num','x_id', 'y_id'] )
            recorded_parquet_df = recorded_parquet.to_pandas()
            recorded_parquet_df = recorded_parquet_df.rename(columns = {'x_id':'col_var_id', 'y_id':'target_var_id'})

        # Process each CSV
        time_start = time.time()
        for fpath in csvs:

            time_2start = time.time()
            fname = fpath.name
            if 'registry' in fname:
                continue
            surr_label = fname.split('__')[-1].rsplit('.', 1)[0]
            non_surr_part = fname.rsplit('__', 1)[0]

            pat = rf"(\d+)_E{E}_tau{tau}_lag_(-?\d+)"
            mfile = re.fullmatch(pat, non_surr_part)
            if not mfile:
                pat2 = rf"(\d+)_E{E}_tau{tau}_lag(-?\d+)"
                mfile = re.fullmatch(pat2, non_surr_part)
            if not mfile:
                pat3 = rf"(\d+)_E{E}_tau{tau}_lag{lag}__(neither0)\.csv"
                mfile = re.fullmatch(pat3, fname)
            if not mfile:
                print(f"\tSkipping unrecognized file name {fname}", file=sys.stderr, flush=True)
                continue

            pset_id = mfile.group(1)
            surr_var, surr_num = _parse_surr(surr_label, col_var, target_var)
            if write_path_file_valid is True:
                surr_df_reduced = recorded_parquet_df[(recorded_parquet_df['surr_var']==surr_var) & (recorded_parquet_df['surr_num']==surr_num)]
                if len(surr_df_reduced) > 0:
                    time_2end = time.time()
                    print(f"\tSkipping {fname} because surr_var={surr_var}, surr_num={surr_num} already in {Path(write_path).name}, {time_2end - time_2start:.2f} seconds", file=sys.stdout, flush=True)
                    sub_existing.append(fpath)
                    continue

            grp_d = {'E': E, 'tau': tau, 'lag': lag, 'Tp': Tp, 'knn': knn, 'surr_var': surr_var, 'surr_num': surr_num,
                     'col_var_id': col_var_id,
                     'target_var_id': target_var_id}  # ,'col_var': col_var, 'target_var': target_var,'pset_id': pset_id,'x_age_model_ind': x_age_model_ind,'y_age_model_ind': y_age_model_ind}
            skip = False
            print('\tgrp_d for existence check:', grp_d, file=sys.stdout, flush=True)

            print(f"\tr\tReading {fpath}", file=sys.stdout, flush=True)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                print(f"x\tSkipping unreadable {fpath}", file=sys.stderr, flush=True)
                continue

            # pull known metrics if present; default to NA
            present = set(df.columns)
            take = {}
            for c in ("rho","MAE","RMSE","LibSize","ind_i"):
                take[c] = df[c] if c in present else pd.Series([pd.NA]*len(df))

            if "relation" in present:
                rel_series = df["relation"].astype("string").str.strip().str.replace("  ", " ", regex=False)
            elif "relation_s" in present:
                rel_series = df["relation_s"].astype("string").str.strip().str.replace("  ", " ", regex=False)
            else:
                rel_series = pd.Series(pd.NA, index=df.index, dtype="string")

            # 2) forcing/responding: pass through if present, else try to parse from relation text
            if "forcing" in present and "responding" in present:
                forcing_series = df["forcing"].astype("string").str.strip()
                responding_series = df["responding"].astype("string").str.strip()
            else:
                # parse "X causes Y", "X influences Y", "X -> Y", "X → Y"
                # returns two columns lhs/rhs or NA if it can't parse
                parsed = rel_series.str.extract(
                    r"^\s*(?P<lhs>.+?)\s*(?:causes|influences|->|→)\s*(?P<rhs>.+?)\s*$",
                    flags=re.IGNORECASE
                )
                forcing_series = parsed["lhs"].astype("string")
                responding_series = parsed["rhs"].astype("string")

            # ----------------------------------------------------------------------

            fixed = {
                "E": E,
                "tau": tau,
                "Tp": Tp,
                "lag": lag,
                "knn": knn if knn is not None else pd.NA,
                "pset_id": str(pset_id),
                "surr_var": surr_var,
                "surr_num": int(surr_num),
                "x_id": config.col.var_id,
                "x_age_model_ind": x_age_model_ind if x_age_model_ind is not None else pd.NA,
                "x_var": config.col.var,
                "y_id": config.target.var_id,
                "y_age_model_ind": y_age_model_ind if y_age_model_ind is not None else pd.NA,
                "y_var": config.target.var,
            }
            fixed_df = pd.DataFrame({k: [v] * len(df) for k, v in fixed.items()})
            out = pd.concat([fixed_df, pd.DataFrame(take)], axis=1)
            out["relation"] = rel_series
            out["forcing"] = forcing_series
            out["responding"] = responding_series

            # optional provenance passthrough (unchanged)
            for c in ("code_version", "align_method", "interp_method", "started_at", "finished_at", "status"):
                if c in present:
                    out[c] = df[c].astype(str)

            records.append(out)
            time_2end = time.time()
            print(f"\tProcessed {fname} with {len(df)} rows in {time_2end - time_2start:.2f} seconds", file=sys.stdout, flush=True)

        time_end = time.time()
        print(f"\tCompleted reading {len(csvs)} files under E{E}_tau{tau}, lag={lag} in {time_end - time_start:.2f} seconds", file=sys.stdout, flush=True)

        if len(records)==0:
            print(f"\tNo new CSV rows discovered under {e_tau_dir_read}, lag={lag}; existing={len(sub_existing)}", file=sys.stdout, flush=True)
            existing.append(write_path)
            continue

        res = pd.concat(records, ignore_index=True)

        res["uid"] = res.apply(_make_uid, axis=1)

        # light typing
        for c in ("E","tau","Tp","lag","LibSize","surr_num","x_age_model_ind","y_age_model_ind"):
            if c in res.columns:
                res[c] = pd.to_numeric(res[c], errors="coerce").astype("Int64")
        new_table = pa.Table.from_pandas(res, preserve_index=False)

        if write_path.exists() is True:
            existing_table = pq.read_table(write_path)
            print('Existing rows in', write_path, ':', existing_table.num_rows, file=sys.stdout, flush=True)
            new_table = pa.concat_tables([existing_table, new_table], promote=True)

            existing_table=None
            gc.collect()
            pa.default_memory_pool().release_unused()

        print('\tCombined rows:', new_table.num_rows, file=sys.stdout, flush=True)
        print('\t', write_path, file=sys.stdout, flush=True)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table,
                       write_path, compression="zstd", use_dictionary=True)

        new_table = None
        gc.collect()
        pa.default_memory_pool().release_unused()

        write_paths.append(write_path)

    return write_paths, existing




if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    second_suffix = ''
    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}'

    calc_location = set_calc_path(args, proj_dir, config, second_suffix)
    output_dir = set_output_path(args, calc_location, config)

    calc_grps_csv = calc_location / check_csv(config.csvs.calc_grps)
    calc_grps_df = pd.read_csv(calc_grps_csv)
    E_tau_grp_csv = args.parameters if args.parameters is not None else config.csvs.e_tau_grps
    if args.parameters is not None:
        print('Using E_tau groups from', args.parameters, file=sys.stdout, flush=True)
    else:
        print('Using E_tau groups from config:', config.csvs.e_tau_grps, file=sys.stdout, flush=True)

    try:
        E_tau_grps = pd.read_csv(calc_location / check_csv(E_tau_grp_csv))
    except:
        E_tau_grps = pd.DataFrame()

    if len(E_tau_grps) > 0:
        if args.inds is not None:
            ind = int(args.inds[-1])
            try:
                E_tau_grp_d = E_tau_grps.iloc[ind].to_dict()
            except Exception as e:
                print('E_tau_grp_d error:', e, file=sys.stderr, flush=True)
                sys.exit(0)

            query_str = ' and '.join([f'{k} == {repr(v)}' for k, v in E_tau_grp_d.items()])
            calc_grps_df2 = calc_grps_df.query(query_str).reset_index(drop=True)
            print(f"Filtered calc_grps_df to {len(calc_grps_df2)} rows matching {E_tau_grp_d}", file=sys.stdout, flush=True)

            for ind2, calc_grp in calc_grps_df2.iterrows():
                calc_grp_d = calc_grp.to_dict()
                print(f"\tcalc_grp {ind}", calc_grp_d, file=sys.stdout, flush=True)
                try:
                    write_paths, existing_paths = package_calc_grp_results_to_parquet(
                        **setup_conversion_from_calc_grp(output_dir, config, calc_grp_d))
                except Exception as e:
                    print('grp error:', e)
    else:
        if args.inds is not None:
            ind = int(args.inds[-1])
            calc_grp_d = calc_grps_df.iloc[ind].to_dict()
            try:
                write_paths, existing = package_calc_grp_results_to_parquet(
                    **setup_conversion_from_calc_grp(output_dir, config, calc_grp_d))
            except Exception as e:
                print('grp error:', e)
        else:
            existing, writes = [], []
            for ind, calc_grp in calc_grps_df.iterrows():
                calc_grp_d = calc_grp.to_dict()
                print(f"calc_grp {ind}", calc_grp_d, file=sys.stdout, flush=True)
                try:
                    write_paths, existing_paths = package_calc_grp_results_to_parquet(
                        **setup_conversion_from_calc_grp(output_dir, config, calc_grp_d))
                    existing.extend(existing_paths)
                    writes.extend(write_paths)
                except Exception as e:
                    print('grp error:', e, file=sys.stderr, flush=True)

