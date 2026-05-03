import sys
import time
from collections import defaultdict
from pathlib import Path
import os
import re
import gc
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.routing import template_replace, parse_surr_label
    from cedarkit.utils.tables import drop_duplicates, make_uid
    from cedarkit.utils.cli import setup_logging, log_line
    from cedarkit.utils.routing.paths import (
        resolve_consolidated_dir,
        resolve_intermediate_dir,
        set_output_path,
    )
except ImportError:
    from utils.routing.file_name_parsers import template_replace, parse_surr_label
    from utils.tables.parquet_tools import drop_duplicates, make_uid
    from utils.cli.logging import setup_logging, log_line
    from utils.routing.paths import (
        _resolve_consolidated_dir,
        _resolve_intermediate_dir,
        set_output_path,
    )


def setup_conversion_from_calc_grp(calc_location, config, calc_grp_d, output_dir = None, intermediate_type = 'csv', consolidated_type='parquet'):
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
    # try:
    #     output_sub = config.output.dir
    # except:
    #     output_sub = 'parquet'

    # directory of existing CSV input


    # update parts_d as determined from file structure with values from config (in case they differ)
    parts_d['col_var_id']=config.col.var_id
    parts_d['target_var_id']=config.target.var_id

    intermediate_output = resolve_intermediate_dir(calc_location, config, intermediate_type)
    e_tau_dir_read = intermediate_output / template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)
    legacy_output = None
    legacy_dir = None

    # Backward-compat: some runs wrote CSVs under calc_refactor (output_dir) instead of intermediate/csv.
    # If intermediate is missing for this group, fall back to legacy output root.
    if (not e_tau_dir_read.exists()) or (not e_tau_dir_read.is_dir()):
        legacy_output = output_dir or set_output_path(None, Path(calc_location), config)
        legacy_dir = Path(legacy_output) / template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)
        if legacy_dir.exists() and legacy_dir.is_dir():
            print(
                f"intermediate path missing, falling back to legacy output path: {legacy_dir}",
                file=sys.stdout,
                flush=True,
            )
            e_tau_dir_read = legacy_dir

    print(
        'setup_conversion_from_calc_grp routing:',
        {
            'calc_location': str(calc_location),
            'group': {k: parts_d.get(k) for k in ('col_var_id', 'target_var_id', 'E', 'tau', 'Tp', 'knn', 'lag')},
            'intermediate_output': str(intermediate_output),
            'intermediate_exists': intermediate_output.exists(),
            'candidate_source_dir': str(e_tau_dir_read),
            'candidate_source_exists': e_tau_dir_read.exists(),
            'legacy_output': None if legacy_output is None else str(legacy_output),
            'legacy_dir': None if legacy_dir is None else str(legacy_dir),
            'legacy_dir_exists': None if legacy_dir is None else legacy_dir.exists(),
            'pattern': E_tau_grp_pattern,
        },
        file=sys.stdout,
        flush=True,
    )
    # directory of future parquet output
    consolidated_output_location = resolve_consolidated_dir(calc_location, config, consolidated_type)

    e_tau_dir_write = consolidated_output_location/template_replace(E_tau_grp_pattern, parts_d, return_replaced=False)
    print(
        'setup_conversion_from_calc_grp write target:',
        {
            'consolidated_output_location': str(consolidated_output_location),
            'consolidated_parent_exists': consolidated_output_location.parent.exists(),
            'e_tau_dir_write': str(e_tau_dir_write),
        },
        file=sys.stdout,
        flush=True,
    )

    # pull from config
    col_var = config.col.var
    target_var = config.target.var

    return {'e_tau_source_dir': e_tau_dir_read, 'e_tau_dir_write':e_tau_dir_write, 'parts_d': parts_d, 'col_var': col_var, 'target_var': target_var, 'config': config}


def package_calc_grp_results_to_parquet(
    e_tau_source_dir: Path,
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
        e_tau_source_dir (Path): Directory path to read CSV files from.
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

    print(
        'package_calc_grp_results_to_parquet source:',
        {
            'e_tau_source_dir': str(e_tau_source_dir),
            'exists': e_tau_source_dir.exists(),
            'is_dir': e_tau_source_dir.is_dir(),
            'e_tau_dir_write': str(e_tau_dir_write),
        },
        file=sys.stdout,
        flush=True,
    )
    # if CSV input directory does not exist, return
    if not e_tau_source_dir.exists() or not e_tau_source_dir.is_dir():
        print(f"Source directory {e_tau_source_dir} does not exist or is not a directory", file=sys.stderr, flush=True)
        print(f"Source directory {e_tau_source_dir} does not exist or is not a directory", file=sys.stdout, flush=True)
        return write_paths, existing

    lag_dir = None
    lag_dir_d = defaultdict(list)
    if lag is not None:
        if (e_tau_source_dir / f'lag_{lag}').exists() is True:
            lag_dir = e_tau_source_dir / f'lag_{lag}'
        elif (e_tau_source_dir / f'lag{lag}').exists() is True:
            lag_dir = e_tau_source_dir / f'lag{lag}'

        if lag_dir is not None:
            lag_dir_d[lag] = [lag_dir / fn for fn in os.listdir(lag_dir) if fn.endswith('.csv')]
    else:
        lag_dirs = [entry for entry in sorted(os.listdir(e_tau_source_dir)) if entry.startswith("lag")]
        if len(lag_dirs) == 0:
            print(f"No lag* subdirectories found under source {e_tau_source_dir}", file=sys.stdout, flush=True)
            return write_paths, existing
        else:
            print(f"Found {len(lag_dirs)} lag* subdirectories under source {e_tau_source_dir}",file=sys.stdout, flush=True)
            # e_tau_dir_write.mkdir(exist_ok=True, parents=True)

        # gather CSV files under each lag directory
        for entry in lag_dirs:
            lag_dir = Path(os.path.join(e_tau_source_dir, entry))
            if os.path.isdir(lag_dir) is True:
                lag = int(entry.replace('lag', ''))
                lag_dir_d[lag]+= [lag_dir/fn for fn in os.listdir(lag_dir) if fn.endswith('.csv')]

    e_tau_dir_write.mkdir(exist_ok=True, parents=True)

    # process each lag directory, gathering records checking to see if they have already been added to the target parquet file, finally writing to Parquet
    for lag, csvs in lag_dir_d.items():
        records = []
        sub_existing =[]

        grp_d = parts_d.copy()
        grp_d.update({'lag': lag})

        file_name_parquet = template_replace(config.output.parquet.file_format, grp_d, return_replaced=False)
        write_path = e_tau_dir_write/ f"{file_name_parquet}.parquet"
        write_path_file_valid = os.path.exists(write_path)
        if write_path_file_valid is False:
            print(f"No existing parquet file at {write_path}, will create new one.", file=sys.stdout, flush=True)

        # check existing parquet to see what has been recorded
        if write_path_file_valid is True:
            existing_parquet_table = ds.dataset(str(write_path), format="parquet").to_table()
            print('\texisting_parquet_table rows:', existing_parquet_table.num_rows,existing_parquet_table.schema.names, file=sys.stdout, flush=True)
            recorded_parquet = drop_duplicates(existing_parquet_table, on=['E', 'tau', 'lag', 'Tp', 'knn', 'surr_var', 'surr_num', 'x_id', 'y_id'])
            recorded_parquet_df = recorded_parquet.to_pandas()
            recorded_parquet_df = recorded_parquet_df.rename(columns = {'x_id':'col_var_id', 'y_id':'target_var_id'})
            existing.append(write_path)

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
            parsed_surr = parse_surr_label(surr_label, col_var, target_var)
            if parsed_surr is None:
                print(f"\tSkipping {fname} because surrogate label could not be parsed", file=sys.stderr, flush=True)
                continue
            surr_var, surr_num = parsed_surr
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
                rel_series = df["relation"].astype("string").str.replace("  ", " ", regex=False).str.strip()
            elif "relation_s" in present:
                rel_series = df["relation_s"].astype("string").str.replace("  ", " ", regex=False).str.strip()
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
            print(f"\tNo new CSV rows discovered under {e_tau_source_dir}, lag={lag}; existing={len(sub_existing)}", file=sys.stdout, flush=True)
            existing.append(write_path)
            continue

        res = pd.concat(records, ignore_index=True)

        res["uid"] = res.apply(make_uid, axis=1)

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
