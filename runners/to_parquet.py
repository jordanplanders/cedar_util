from pathlib import Path
import time
import sys
import pandas as pd
import os

try:
    from cedarkit.utils.cli import get_parser
    from cedarkit.core.project_config import load_config
    from cedarkit.utils.io import setup_conversion_from_calc_grp, package_calc_grp_results_to_parquet
    from cedarkit.utils.routing import set_calc_path
    from cedarkit.utils.routing.paths import resolve_consolidated_dir
    from cedarkit.utils.routing import check_csv
except ImportError:
    from utils.cli.arg_parser import get_parser
    from core.project_config import load_config
    from utils.io.parquet import setup_conversion_from_calc_grp, package_calc_grp_results_to_parquet
    from utils.routing.paths import set_calc_path
    from utils.routing.paths import _resolve_consolidated_dir
    from utils.routing.file_name_parsers import check_csv
    # Fallback: imports when running as a package
    # from utils.cli.arg_parser import get_parser
    # from core.project_config import load_config
    # from utils.io.parquet_tools import drop_duplicates, _make_uid
    # import utils.routing.paths
    # from utils.routing.file_name_parsers import parse_surr_label, template_replace
    # from utils.routing.paths import set_calc_path, set_output_path


# from tmp_utils.path_utils import set_calc_path, set_output_path, template_replacement

# def check_csv(output_file_name):
#     if '.csv' not in output_file_name:
#         output_file_name = f'{output_file_name}.csv'
#     return output_file_name


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

# drop duplicates in parquet table


# def get_col_var_and_target_var(config, parts_d):
#     col_var = config.get_dynamic_attr("{var}.var", parts_d['col_var_id'])
#     target_var = config.get_dynamic_attr("{var}.var", parts_d['target_var_id'])
#     return col_var, target_var


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    if args.proj_dir is not None:
        proj_dir = Path(args.proj_dir) / proj_name
    else:
        proj_dir = Path(os.getcwd()) / proj_name
    print('Project directory:', proj_dir, file=sys.stdout, flush=True)

    config = load_config(proj_dir / 'proj_config.yaml')

    second_suffix = ''
    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}'

    calc_location = set_calc_path(args, proj_dir, config, second_suffix)
    output_dir = resolve_consolidated_dir(calc_location, config, "parquet")
    print('output_dir:', output_dir, file=sys.stdout, flush=True)

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
            dedup_keys = [k for k in ['E', 'tau', 'knn', 'Tp', 'lag'] if k in calc_grps_df2.columns]
            if len(dedup_keys) > 0:
                before = len(calc_grps_df2)
                calc_grps_df2 = calc_grps_df2.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
                print(
                    f"Deduped calc_grps_df2 on {dedup_keys}: {before} -> {len(calc_grps_df2)} rows",
                    file=sys.stdout,
                    flush=True,
                )
            print(f"Filtered calc_grps_df to {len(calc_grps_df2)} rows matching {E_tau_grp_d}", file=sys.stdout, flush=True)

            for ind2, calc_grp in calc_grps_df2.iterrows():
                calc_grp_d = calc_grp.to_dict()
                print(f"\tcalc_grp {ind}", calc_grp_d, file=sys.stdout, flush=True)
                try:
                    write_paths, existing_paths = package_calc_grp_results_to_parquet(
                        **setup_conversion_from_calc_grp(calc_location, config, calc_grp_d, consolidated_type='parquet', intermediate_type = 'csv'))
                    print(f"\t\tWrote to {write_paths}, existing paths: {existing_paths}", file=sys.stdout, flush=True)
                except Exception as e:
                    print('grp error:', e, file=sys.stderr, flush=True)
    else:
        if args.inds is not None:
            ind = int(args.inds[-1])
            calc_grp_d = calc_grps_df.iloc[ind].to_dict()
            try:
                write_paths, existing = package_calc_grp_results_to_parquet(
                    **setup_conversion_from_calc_grp(calc_location, config, calc_grp_d, consolidated_type='parquet', intermediate_type = 'csv'))
                print(f"Wrote to {write_paths}, existing paths: {existing}", file=sys.stdout, flush=True)
            except Exception as e:
                print('grp error:', e, file=sys.stderr, flush=True)
        else:
            existing, writes = [], []
            dedup_keys = [k for k in ['E', 'tau', 'knn', 'Tp', 'lag'] if k in calc_grps_df.columns]
            if len(dedup_keys) > 0:
                before = len(calc_grps_df)
                calc_grps_df = calc_grps_df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
                print(
                    f"Deduped calc_grps_df on {dedup_keys}: {before} -> {len(calc_grps_df)} rows",
                    file=sys.stdout,
                    flush=True,
                )
            for ind, calc_grp in calc_grps_df.iterrows():
                calc_grp_d = calc_grp.to_dict()
                print(f"calc_grp {ind}", calc_grp_d, file=sys.stdout, flush=True)
                try:
                    write_paths, existing_paths = package_calc_grp_results_to_parquet(
                        **setup_conversion_from_calc_grp(calc_location, config, calc_grp_d, consolidated_type='parquet', intermediate_type = 'csv'))
                    existing.extend(existing_paths)
                    writes.extend(write_paths)
                except Exception as e:
                    print('grp error:', e, file=sys.stderr, flush=True)
