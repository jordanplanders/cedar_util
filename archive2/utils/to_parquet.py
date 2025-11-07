import os, re, json, hashlib, sys
from collections import defaultdict
from pathlib import Path
from itertools import product
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import gc

# from data_objects import extract_from_pattern
from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.location_helpers import *
from utils.data_access import *

from data_obj.data_objects import DataGroup, check_existance_in_table
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
    token examples: 'neither0', 'TSI147', 'temp033', 'both12'
    Returns (surr_var, surr_num) where surr_var ∈ {'neither', x_var, y_var, 'both'}
    """

    # m = re.fullmatch(r"([A-Za-z0-9]+)(\d+)", token)
    # if not m:
    #     return "neither", 0
    # label, num = m.group(1), int(m.group(2))
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


    # m = re.fullmatch(r"([A-Za-z0-9]+)(\d+)", token)
    # if not m:
    #     return "neither", 0
    # label, num = m.group(1), int(m.group(2))
    # lab_l = label.lower()
    # xv, yv = x_var.lower(), y_var.lower()
    #
    # if lab_l == "neither":
    #     return "neither", 0
    # if lab_l == "both":
    #     return "both", num
    # if lab_l == xv:
    #     return x_var, num
    # if lab_l == yv:
    #     return y_var, num
    # # Fallback: treat unknown label as 'neither' but keep the number
    # return "neither", num

def _make_uid(row: pd.Series) -> str:
    blob = json.dumps({k: (None if pd.isna(row[k]) else row[k]) for k in KEY_COLS},
                      sort_keys=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


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


def get_col_var_and_target_var(config, parts_d):
    col_var = config.get_dynamic_attr("{var}.var", parts_d['col_var_id'])
    target_var = config.get_dynamic_attr("{var}.var", parts_d['target_var_id'])
    return col_var, target_var


def setup_conversion_from_calc_grp(output_dir, config, calc_grp_d):

    # find calc_grp_d via index and convert to dict (parts_d)
    parts_d = calc_grp_d  #calc_grps_df.iloc[ind].to_dict()
    # construct path pattern
    fallback_E_tau_grp_pattern = 'knn_{knn}/tp_{Tp}/{col_var_id}_{target_var_id}/E{E}_tau{tau}'
    E_tau_grp_pattern = config.output.dir_structure if config is not None else fallback_E_tau_grp_pattern
    #template replacement
    e_tau_dir_read = output_dir/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)
    e_tau_dir_write = output_dir/'parquet'/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)

    # pull from config, will need to alter once move to var yamls or var objects
    col_var, target_var = get_col_var_and_target_var(config, parts_d)

    return {'e_tau_dir_read': e_tau_dir_read, 'e_tau_dir_write':e_tau_dir_write, 'parts_d': parts_d, 'col_var': col_var, 'target_var': target_var, 'config': config}

def setup_conversion_from_e_tau_dir(config, e_tau_dir):

    # construct path pattern
    E_tau_grp_pattern = config.output.dir_structure if config is not None else 'knn_{knn}/tp_{Tp}/{col_var_id}_{target_var_id}/E{E}_tau{tau}'
    sub_path = str(e_tau_dir).split('calc_refactor')[-1].lstrip('/').lstrip('\\')
    # extract parts_d from path
    parts_d = extract_from_pattern(sub_path, E_tau_grp_pattern)

    # pull from config, will need to alter once move to var yamls or var objects
    col_var, target_var = get_col_var_and_target_var(config, parts_d)

    e_tau_dir_read = Path(e_tau_dir)
    e_tau_dir_write = output_dir/'parquet'/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)

    return {'e_tau_dir_read': e_tau_dir_read, 'e_tau_dir_write':e_tau_dir_write,'parts_d':parts_d, 'col_var':col_var, 'target_var':target_var, 'config':config}


def package_calc_grp_results_to_parquet(
    e_tau_dir_read: Path,
    e_tau_dir_write: Path,
    parts_d: str,
    col_var: str,            # e.g., 'temp'
    target_var: str,         # e.g., 'TSI'
    # Tp: int = 1,
    config: dict | None = None,
    # knn: int | None = None,  # if set, add knn column with this value
    x_age_model_ind: int | None = None,  # integer pointers from YAML
    y_age_model_ind: int | None = None,
    # relation_label: str | None = None,
    write_path: str | None = None,
        overwrite: bool = True,
        # pattern: str | None = None,
):
    # currently parts_d is expected to be the content of a calc_grp, but is parsed and reformed into grp_d rather than as a pass through


    E = parts_d.get('E', None)
    tau = parts_d.get('tau', None)
    Tp = parts_d.get('Tp', 1)
    knn = parts_d.get('knn', None)
    col_var_id = parts_d.get('col_var_id', None)
    target_var_id = parts_d.get('target_var_id', None)

    existing = []
    write_paths = []
    if not e_tau_dir_read.exists() or not e_tau_dir_read.is_dir():
        print(f"Directory {e_tau_dir_read} does not exist or is not a directory", file=sys.stderr, flush=True)
        print(f"Directory {e_tau_dir_read} does not exist or is not a directory", file=sys.stdout, flush=True)
        return write_paths, existing

    lag_dirs = [entry for entry in sorted(os.listdir(e_tau_dir_read)) if entry.startswith("lag")]
    if len(lag_dirs) == 0:
        print(f"No lag* subdirectories found under {e_tau_dir_read}", file=sys.stdout, flush=True)
        return write_paths, existing
    else:
        print(f"Found {len(lag_dirs)} lag* subdirectories under {e_tau_dir_read}",file=sys.stdout, flush=True)
        e_tau_dir_write.mkdir(exist_ok=True, parents=True)

    lag_dir_d = defaultdict(list)
    for entry in lag_dirs:
        lag_dir = Path(os.path.join(e_tau_dir_read, entry))
        if os.path.isdir(lag_dir) is True:
            mlag = re.fullmatch(r"lag(-?\d+)", entry)
            lag = int(mlag.group(1))
            lag_dir_d[lag]+= [lag_dir/fn for fn in os.listdir(lag_dir) if fn.endswith('.csv')]

            # except Exception as e:
            #     print(f"Skipping unreadable lag directory {lag_dir}: {e}", file=sys.stderr, flush=True)

    for lag, csvs in lag_dir_d.items():
        records = []
        sub_existing =[]
        write_path = e_tau_dir_write/ f"E{E}_tau{tau}_lag{lag}.parquet"
        write_path_file_valid = os.path.exists(write_path)
        if write_path_file_valid is True:
            grp_d = parts_d.copy()
            grp_d.update({'lag': lag})
            do = DataGroup(grp_d)
            try:
                existing_parquet = ds.dataset(str(write_path), format="parquet")
                groupconfig_file, filtered_table = do._internal_query(existing_parquet)
            except:
                write_path_file_valid = False

        # if len(csvs)>0:
        for fpath in csvs:#sorted(os.listdir(lag_dir)):
            fname = fpath.name
            # {pset}_E{E}_tau{tau}_lag{lag}__{suffix}.csv
            surr_label = fname.split('__')[-1].rsplit('.', 1)[0]
            non_surr_part = fname.rsplit('__', 1)[0]
            # pat = rf"(\d+)_E{E}_tau{tau}_lag{lag}__([A-Za-z]+[0-9]+)\.csv"
            pat = rf"(\d+)_E{E}_tau{tau}_lag{lag}"#__([A-Za-z0-9]+[0-9]+)\.csv"
            mfile = re.fullmatch(pat, non_surr_part)
            if not mfile:
                # allow '__neither0.csv' explicitly
                mfile = re.fullmatch(rf"(\d+)_E{E}_tau{tau}_lag{lag}__(neither0)\.csv", fname)
            if not mfile:
                continue
            # pset_id, suffix = mfile.group(1), mfile.group(2)

            pset_id = mfile.group(1)
            surr_var, surr_num = _parse_surr(surr_label, col_var, target_var)
            grp_d = {'E': E, 'tau': tau, 'lag': lag, 'Tp': Tp, 'knn': knn, 'surr_var': surr_var, 'surr_num': surr_num,
                     'col_var_id': col_var_id,
                     'target_var_id': target_var_id}  # ,'col_var': col_var, 'target_var': target_var,'pset_id': pset_id,'x_age_model_ind': x_age_model_ind,'y_age_model_ind': y_age_model_ind}
            skip = False

            if write_path_file_valid is True:
                filtered_table2 = check_existance_in_table(filtered_table, grp_d)
                if (filtered_table2 is not False) and (filtered_table2 is not None):
                    try:
                        if filtered_table2.num_rows>0:
                            skip = True
                    except Exception as e:
                        print(f"Error checking existing table {write_path} for {grp_d}: {e}", file=sys.stderr, flush=True)

            if skip is True:
                print(f"s\tSkipping {fname} because already in {Path(write_path).name}", file=sys.stdout, flush=True)
                sub_existing.append(fpath)
                continue

            print(f"r\tReading {fpath}", file=sys.stdout, flush=True)
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
                "x_id": col_var_id,
                "x_age_model_ind": x_age_model_ind if x_age_model_ind is not None else pd.NA,
                "x_var": col_var,
                "y_id": target_var_id,
                "y_age_model_ind": y_age_model_ind if y_age_model_ind is not None else pd.NA,
                "y_var": target_var,
            }
            fixed_df = pd.DataFrame({k: [v] * len(df) for k, v in fixed.items()})
            # print('fixed_df', fixed_df.head())
            out = pd.concat([fixed_df, pd.DataFrame(take)], axis=1)
            # print('out before extras', out.head())
            # attach row-wise relation/forcing/responding (may contain NA if not present/parsable)
            out["relation"] = rel_series
            out["forcing"] = forcing_series
            out["responding"] = responding_series

            # optional provenance passthrough (unchanged)
            for c in ("code_version", "align_method", "interp_method", "started_at", "finished_at", "status"):
                if c in present:
                    out[c] = df[c].astype(str)

            records.append(out)

        if not records:
            print(f"No new CSV rows discovered under {e_tau_dir_read}, lag={lag}; existing={len(sub_existing)}", file=sys.stdout, flush=True)
            existing.append(write_path)
            continue

        res = pd.concat(records, ignore_index=True)

        # stable uid includes age_model_ind and surr_var
        res["uid"] = res.apply(_make_uid, axis=1)

        # light typing
        for c in ("E","tau","Tp","lag","LibSize","surr_num","x_age_model_ind","y_age_model_ind"):
            if c in res.columns:
                res[c] = pd.to_numeric(res[c], errors="coerce").astype("Int64")
        # print(f"\tFound {len(res)} CSV rows under E{E}_tau{tau}, lag={lag} from {count} files")
        # write beside the E*_tau* folder by default
        # print('res columns', res.columns)
        new_table = pa.Table.from_pandas(res, preserve_index=False)

        if write_path_file_valid is True:
            existing_table = pq.read_table(write_path)
            print('Existing rows in', write_path, ':', existing_table.num_rows, file=sys.stdout, flush=True)
            new_table = pa.concat_tables([existing_table, new_table], promote=True)

            existing_table=None
            gc.collect()
            pa.default_memory_pool().release_unused()

        print('\tCombined rows:', new_table.num_rows, file=sys.stdout, flush=True)
        print('\t', write_path, file=sys.stdout, flush=True)
        pq.write_table(new_table,
                       write_path, compression="zstd", use_dictionary=True)

        new_table = None
        gc.collect()
        pa.default_memory_pool().release_unused()

        # df = pd.read_parquet(write_path, columns=["E","tau", "relation", "forcing", "responding", 'lag', 'surr_var']).drop_duplicates()
        # # print('checking surr var', df.surr_var.unique())
        # print("\tmissing forcing/responding:",
        #       df["forcing"].isna().sum(), df["responding"].isna().sum(), df["surr_var"].isna().sum())
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

    # load calc_grp csv
    # calc_grps_csv = proj_dir/config.calc_carc_dir.name/config.calc_carc_dir.csvs.calc_grps
    calc_grps_csv = calc_location / check_csv(config.calc_carc_dir.csvs.calc_grp_csv)
    calc_grps_df = pd.read_csv(calc_grps_csv)

    if args.parameters is not None:
        print('Using E_tau groups from', args.parameters, file=sys.stdout, flush=True)
        E_tau_grps = pd.read_csv(calc_location / check_csv(args.parameters))
        if args.inds is not None:
            ind = int(args.inds[-1])
            try:
                E_tau_grp_d = E_tau_grps.iloc[ind].to_dict()
            except Exception as e:
                print('E_tau_grp_d error:', e, file=sys.stderr, flush=True)
                sys.exit(0)
            # calc_grps_df = calc_grps_df[calc_grps_df.isin(E_tau_grp_d).all(axis=1)].reset_index(drop=True)
            # Suppose E_tau_grp_d = {'E': 4, 'tau': 1}
            query_str = ' and '.join([f'{k} == {repr(v)}' for k, v in E_tau_grp_d.items()])
            calc_grps_df2 = calc_grps_df.query(query_str).reset_index(drop=True)
            print(f"Filtered calc_grps_df to {len(calc_grps_df2)} rows matching {E_tau_grp_d}", file=sys.stdout, flush=True)
            for ind2, calc_grp in calc_grps_df2.iterrows():
                calc_grp_d = calc_grp.to_dict()
                print(f"calc_grp {ind}", calc_grp_d, file=sys.stdout, flush=True)
                # print('existing:', len(existing), 'written:', len(write_paths))
                try:
                    write_paths, existing_paths = package_calc_grp_results_to_parquet(
                        **setup_conversion_from_calc_grp(output_dir, config, calc_grp_d))
                    # existing.extend(existing_paths)
                    # write_paths.extend(write_paths)
                except Exception as e:
                    print('grp error:', e)
    else:
        print()
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




# def package_e_tau_results_to_parquet(
#     e_tau_dir: str,
#     col_var_id: str,
#     target_var_id: str,
#     col_var: str,            # e.g., 'temp'
#     target_var: str,         # e.g., 'TSI'
#     Tp: int = 1,
#     config: dict | None = None,
#     knn: int | None = None,  # if set, add knn column with this value
#     x_age_model_ind: int | None = None,  # integer pointers from YAML
#     y_age_model_ind: int | None = None,
#     # relation_label: str | None = None,
#     write_path: str | None = None,
#         overwrite: bool = True,
#         # pattern: str | None = None,
# ):
#     """
#     Reads all lags under one E*_tau* directory and writes a single results.parquet.
#
#     Expects filenames like:
#       {pset}_E{E}_tau{tau}_lag{lag}__{SUFFIX}.csv
#     where SUFFIX ∈ {'neither0','both12', f'{x_var}033', f'{y_var}147', ...}
#     """
#     base = os.path.basename(e_tau_dir)  # e.g., 'E4_tau5'
#
#     fallback_E_tau_grp_pattern = 'knn_{knn}/tp_{Tp}/{col_var_id}_{target_var_id}/E{E}_tau{tau}'
#     E_tau_grp_pattern = config.output.dir_structure if config is not None else fallback_E_tau_grp_pattern
#     sub_path = str(e_tau_dir).split('calc_refactor')[-1].lstrip('/').lstrip('\\')
#     parts_d = extract_from_pattern(sub_path, E_tau_grp_pattern)
#     # going to load in parts_d, e_tau_dir, col_var, target_var
#
#     E = parts_d.get('E', None)
#     tau = parts_d.get('tau', None)
#     Tp = parts_d.get('Tp', None)
#     knn = parts_d.get('knn', None)
#     col_var_id = parts_d.get('col_var_id', None)
#     target_var_id = parts_d.get('target_var_id', None)
#
#     existing = []
#     write_paths = []
#     lag_dirs = [entry for entry in sorted(os.listdir(e_tau_dir)) if entry.startswith("lag")]
#     if len(lag_dirs) == 0:
#         print(f"No lag* subdirectories found under {e_tau_dir}")
#         return write_paths, existing
#     else:
#         print(f"Found {len(lag_dirs)} lag* subdirectories under {e_tau_dir}")
#
#     for entry in lag_dirs:
#         lag_dir = os.path.join(e_tau_dir, entry)
#         if not os.path.isdir(lag_dir):
#             continue
#         mlag = re.fullmatch(r"lag(-?\d+)", entry)
#         if not mlag:
#             continue
#         lag = int(mlag.group(1))
#
#         write_path = os.path.join(e_tau_dir, f"E{E}_tau{tau}_lag{lag}.parquet")
#
#         records = []
#         count = 0
#         for fname in sorted(os.listdir(lag_dir)):
#             if not fname.endswith(".csv"):
#                 continue
#             count += 1
#             # {pset}_E{E}_tau{tau}_lag{lag}__{suffix}.csv
#             surr_label = fname.split('__')[-1].rsplit('.', 1)[0]
#             non_surr_part = fname.rsplit('__', 1)[0]
#             # pat = rf"(\d+)_E{E}_tau{tau}_lag{lag}__([A-Za-z]+[0-9]+)\.csv"
#             pat = rf"(\d+)_E{E}_tau{tau}_lag{lag}"#__([A-Za-z0-9]+[0-9]+)\.csv"
#             mfile = re.fullmatch(pat, non_surr_part)
#             if not mfile:
#                 # allow '__neither0.csv' explicitly
#                 mfile = re.fullmatch(rf"(\d+)_E{E}_tau{tau}_lag{lag}__(neither0)\.csv", fname)
#             if not mfile:
#                 continue
#             # pset_id, suffix = mfile.group(1), mfile.group(2)
#
#             pset_id = mfile.group(1)
#             surr_var, surr_num = _parse_surr(surr_label, col_var, target_var)
#             grp_d = {'E': E, 'tau': tau, 'lag': lag, 'Tp': Tp, 'knn': knn, 'surr_var': surr_var, 'surr_num': surr_num,
#                      'col_var_id': col_var_id,
#                      'target_var_id': target_var_id}  # ,'col_var': col_var, 'target_var': target_var,'pset_id': pset_id,'x_age_model_ind': x_age_model_ind,'y_age_model_ind': y_age_model_ind}
#             # print(grp_d)
#             skip = False
#             if os.path.exists(write_path):
#                 do = DataGroup(grp_d)
#                 groupconfig_file, filtered_table = do._internal_query(ds.dataset(str(write_path), format="parquet"))
#                 if filtered_table.num_rows>0:
#                     skip = True
#
#             # if os.path.exists(write_path) and not overwrite:
#             #     existing.append(write_path)
#             #     # print(f"Skipping E{E}_tau{tau}_lag{lag} because {write_path} exists and overwrite=False")
#             #     continue
#             #     # raise FileExistsError(f"{write_path} exists and overwrite=False")
#
#             if skip is True:
#                 print(f"s\tSkipping {fname} because already in {Path(write_path).name}")
#                 continue
#
#             fpath = os.path.join(lag_dir, fname)
#             try:
#                 df = pd.read_csv(fpath)
#             except Exception:
#                 print(f"x\tSkipping unreadable {fpath}")
#                 continue
#
#             # pull known metrics if present; default to NA
#             present = set(df.columns)
#             take = {}
#             for c in ("rho","MAE","RMSE","LibSize","ind_i"):
#                 take[c] = df[c] if c in present else pd.Series([pd.NA]*len(df))
#
#             if "relation" in present:
#                 rel_series = df["relation"].astype("string").str.strip().str.replace("  ", " ", regex=False)
#             elif "relation_s" in present:
#                 rel_series = df["relation_s"].astype("string").str.strip().str.replace("  ", " ", regex=False)
#             else:
#                 rel_series = pd.Series(pd.NA, index=df.index, dtype="string")
#
#             # 2) forcing/responding: pass through if present, else try to parse from relation text
#             if "forcing" in present and "responding" in present:
#                 forcing_series = df["forcing"].astype("string").str.strip()
#                 responding_series = df["responding"].astype("string").str.strip()
#             else:
#                 # parse "X causes Y", "X influences Y", "X -> Y", "X → Y"
#                 # returns two columns lhs/rhs or NA if it can't parse
#                 parsed = rel_series.str.extract(
#                     r"^\s*(?P<lhs>.+?)\s*(?:causes|influences|->|→)\s*(?P<rhs>.+?)\s*$",
#                     flags=re.IGNORECASE
#                 )
#                 forcing_series = parsed["lhs"].astype("string")
#                 responding_series = parsed["rhs"].astype("string")
#
#             # ----------------------------------------------------------------------
#
#             fixed = {
#                 "E": E,
#                 "tau": tau,
#                 "Tp": Tp,
#                 "lag": lag,
#                 "knn": knn if knn is not None else pd.NA,
#                 "pset_id": str(pset_id),
#                 "surr_var": surr_var,
#                 "surr_num": int(surr_num),
#                 "x_id": col_var_id,
#                 "x_age_model_ind": x_age_model_ind if x_age_model_ind is not None else pd.NA,
#                 "x_var": col_var,
#                 "y_id": target_var_id,
#                 "y_age_model_ind": y_age_model_ind if y_age_model_ind is not None else pd.NA,
#                 "y_var": target_var,
#             }
#             fixed_df = pd.DataFrame({k: [v] * len(df) for k, v in fixed.items()})
#             # print('fixed_df', fixed_df.head())
#             out = pd.concat([fixed_df, pd.DataFrame(take)], axis=1)
#             # print('out before extras', out.head())
#             # attach row-wise relation/forcing/responding (may contain NA if not present/parsable)
#             out["relation"] = rel_series
#             out["forcing"] = forcing_series
#             out["responding"] = responding_series
#
#             # optional provenance passthrough (unchanged)
#             for c in ("code_version", "align_method", "interp_method", "started_at", "finished_at", "status"):
#                 if c in present:
#                     out[c] = df[c].astype(str)
#
#             records.append(out)
#
#         if not records:
#             print(f"No CSV rows discovered under {e_tau_dir}, lag={lag}")
#             continue
#
#         res = pd.concat(records, ignore_index=True)
#
#         # stable uid includes age_model_ind and surr_var
#         res["uid"] = res.apply(_make_uid, axis=1)
#
#         # light typing
#         for c in ("E","tau","Tp","lag","LibSize","surr_num","x_age_model_ind","y_age_model_ind"):
#             if c in res.columns:
#                 res[c] = pd.to_numeric(res[c], errors="coerce").astype("Int64")
#         print(f"\tFound {len(res)} CSV rows under E{E}_tau{tau}, lag={lag} from {count} files")
#         # write beside the E*_tau* folder by default
#         # print('res columns', res.columns)
#         new_table = pa.Table.from_pandas(res, preserve_index=False)
#
#         if os.path.exists(write_path):
#             existing_table = pq.read_table(write_path)
#             print('Existing rows in', write_path, ':', existing_table.num_rows)
#             new_table = pa.concat_tables([existing_table, new_table], promote=True)
#
#         print('\tCombined rows:', new_table.num_rows)
#         print('\t', write_path)
#         pq.write_table(new_table,
#                        write_path, compression="zstd", use_dictionary=True)
#
#         df = pd.read_parquet(write_path, columns=["E","tau", "relation", "forcing", "responding", 'lag', 'surr_var']).drop_duplicates()
#         # print('checking surr var', df.surr_var.unique())
#         print("\tmissing forcing/responding:",
#               df["forcing"].isna().sum(), df["responding"].isna().sum())
#         write_paths.append(write_path)
#
#
#     return write_paths, existing