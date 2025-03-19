import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import re
import os
import sys

# Define directories for notebooks and calculations
notebooks_dir = Path(os.getcwd())/'notebooks'
calc_carc = notebooks_dir/'calc_carc'
calc_dir = notebooks_dir/'calc_dir'

# Define a pattern to match percentile difference columns
pattern = re.compile(r'^r\d+p-s\d+p$')
pattern_surr = re.compile(r'^s\d+p$')
pattern_surr2= re.compile(r'^\d+p$')
pattern_real = re.compile(r'^r\d+p$')

# Function to process each group
def process_group_summary(args):
    grp, grp_df = args
    summary_dfs = []
    libsize_summaries = []

    # Extract metadata for the group
    summary_meta = grp_df[['temp_author', 'tsi_author', 'tau', 'E', 'knn', 'Tp', 'lag', 'sample', 'weighted', 'target_var', 'col_var']].iloc[0].to_dict()

    # Filter surrogate data frames
    _surr_dfs = grp_df[grp_df['surr_var'] != 'neither'].copy()

    for surr_var, surr_var_df in _surr_dfs.groupby('surr_var'):
        surr_run_pctile_list = []
        if len(surr_var_df) > 0:
            for pset_id, pset_df in surr_var_df.groupby('pset_id'):
                # Define the path for the percentiles data frame
                ptiles_df_path = Path(pset_df['df_path'].iloc[0]).parent / (str(pset_id) + f'_pctiles_{surr_var}2.csv')
                if ptiles_df_path.exists():
                    surr_run_df_pctile = pd.read_csv(ptiles_df_path, index_col=0)
                    for relation, rel_df in surr_run_df_pctile.groupby('relation'):
                        # Filter rows where the difference between the 5th and 95th percentiles is positive
                        matching_columns_delta = [col for col in rel_df.columns if pattern.match(col)]
                        for delta in matching_columns_delta:
                            surr_grp_df_pos = rel_df[rel_df[delta] > 0]
                            summary_d = summary_meta.copy()
                            if len(surr_grp_df_pos) == 0:
                                summary_d.update({'relation': relation, 'surr_var': surr_var, 'cusp': 0, 'streak_count': 0, 'max_len': 0, 'delta': delta})
                            else:
                                # Calculate the ideal library sizes and streak count
                                ideal_libsizes = np.arange(rel_df.LibSize.max(), surr_grp_df_pos.LibSize.min() - 1, -5)
                                ik = 0
                                cusp = 0
                                end = False
                                missing = []
                                for libsize in ideal_libsizes:
                                    if libsize in surr_grp_df_pos.LibSize.values:
                                        if not end:
                                            ik += 1
                                            cusp = libsize
                                    else:
                                        end = True
                                        missing.append(libsize)

                                summary_d.update({'relation': relation, 'surr_var': surr_var, 'cusp': cusp, 'streak_count': ik, 'max_len': len(ideal_libsizes), 'delta': delta})
                            libsize_summaries.append(summary_d)

                    surr_run_pctile_list.append(surr_run_df_pctile)
                else:
                    print('\tnot exists', ptiles_df_path, file=sys.stderr, flush=True)

            if len(surr_run_pctile_list) > 0:
                # Consolidate percentile data frames
                surr_df_pctile_df_consol = pd.concat(surr_run_pctile_list)
                summary_info = []
                for relation, rel_df in surr_df_pctile_df_consol.groupby('relation'):
                    matching_columns_delta = [col for col in rel_df.columns if pattern.match(col)]

                    for lib_size, _df in rel_df.groupby('LibSize'):
                        num_real = rel_df.n_real.iloc[0]
                        summary_d = summary_meta.copy()
                        summary_d.update({'relation': relation, 'LibSize': lib_size, 'n_real': num_real, 'num_surr': len(surr_run_pctile_list), 'surr_var': surr_var})

                        # Calculate averages of relevant percentiles

                        matching_columns_surr = [col for col in _df.columns if pattern_surr.match(col)]
                        if len(matching_columns_surr) == 0:
                            matching_columns_surr = [col for col in _df.columns if pattern_surr2.match(col)]

                        matching_columns_real = [col for col in _df.columns if pattern_real.match(col)]
                        for col in matching_columns_surr:
                            summary_d[f'avg_{col}'] = _df[col].mean()
                        for col in matching_columns_real:
                            summary_d[f'avg_{col}'] = _df[col].mean()

                        for col in matching_columns_delta:
                            pct_pos = len(_df[_df[col] > 0]) / len(_df)
                            summary_d[f'pct_plus_{col}'] = pct_pos
                        summary_info.append(summary_d)

                if summary_info:
                    summary_dfs.append(pd.DataFrame(summary_info))
                    print('summary_info', len(summary_info), file=sys.stdout, flush=True)
    return summary_dfs, libsize_summaries

if __name__ == '__main__':
    # Read the CSV file with calculation logs
    calc_log2_df = pd.read_csv(calc_carc/'calc_log3.csv', index_col=0)

    # Group by the specified columns
    grouped = calc_log2_df.groupby(['temp_author', 'tsi_author', 'tau', 'E', 'knn', 'Tp', 'lag', 'sample', 'weighted', 'target_var', 'col_var'])

    args = [(grp, grp_df) for grp, grp_df in grouped]

    # Determine the number of CPUs to use from the SLURM environment variable
    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

    # Use multiprocessing to parallelize the process
    with Pool(num_cpus) as pool:
        results = pool.map(process_group_summary, args)

    summary_dfs = []
    libsize_summaries = []

    # Collect results from parallel processing
    for summary_df, libsize_summary in results:
        if summary_df:
            summary_dfs.extend(summary_df)
        if libsize_summary:
            libsize_summaries.extend(libsize_summary)

    # Combine the results into data frames
    summary_df = pd.concat(summary_dfs) if summary_dfs else pd.DataFrame()
    libsize_summaries_df = pd.DataFrame(libsize_summaries)

    # Save the results to CSV files
    summary_df.to_csv(calc_carc / 'summary_df2.csv')
    libsize_summaries_df.to_csv(calc_carc / 'libsize_summaries_df2.csv')



# import pandas as pd
# import numpy as np
# from pathlib import Path
# from multiprocessing import Pool
# import re
# import os
# import sys
#
# # Define directories for notebooks and calculations
# notebooks_dir = Path(os.getcwd())/'notebooks'
# calc_carc = notebooks_dir/'calc_carc'
# calc_dir = notebooks_dir/'calc_dir'
#
# # Define a pattern to match percentile difference columns
# pattern = re.compile(r'^r\d+p-s\d+p$')
# pattern_surr = re.compile(r'^s\d+p$')
# pattern_real = re.compile(r'^r\d+p$')
#
#
# # Function to process each group
# def process_group_summary(args):
#     grp, grp_df = args
#     summary_dfs = []
#     libsize_summaries = []
#
#     # Extract metadata for the group
#     summary_meta = grp_df[['temp_author', 'tsi_author', 'tau', 'E', 'knn', 'Tp', 'lag', 'sample', 'weighted', 'target_var', 'col_var']].iloc[0].to_dict()
#
#     # Filter surrogate data frames
#     _surr_dfs = grp_df[grp_df['surr_var'] != 'neither'].copy()
#
#     for surr_var, surr_var_df in _surr_dfs.groupby('surr_var'):
#         surr_run_pctile_list = []
#         if len(surr_var_df) > 0:
#             for pset_id, pset_df in surr_var_df.groupby('pset_id'):
#                 # Define the path for the percentiles data frame
#                 ptiles_df_path = Path(pset_df['df_path'].iloc[0]).parent / (str(pset_id) + f'_pctiles_{surr_var}.csv')
#                 if ptiles_df_path.exists():
#                     surr_run_df_pctile = pd.read_csv(ptiles_df_path, index_col=0)
#                     for relation, rel_df in surr_run_df_pctile.groupby('relation'):
#                         # Filter rows where the difference between the 5th and 95th percentiles is positive
#                         matching_columns_delta = [col for col in rel_df if pattern.match(col)]
#                         for delta in matching_columns_delta:#['r5p-s95p', 'r10p-s90p', 'r25p-s75p']:
#                             surr_grp_df_pos = rel_df[rel_df[delta] > 0]
#                             summary_d = summary_meta.copy()
#                             if len(surr_grp_df_pos) == 0:
#                                 # surr_grp_df_npos = rel_df[rel_df[delta] <= 0]
#                                 # end_window = 5 # window of 5 libsize increments
#                                 # avg_delta = surr_grp_df_npos[surr_grp_df_npos.LibSize>=rel_df.LibSize.max()-end_window][delta].mean()
#                                 # If no positive differences, set summary statistics to zero
#                                 summary_d.update({'relation': relation, 'surr_var': surr_var, 'cusp': 0, 'streak_count': 0, 'max_len': 0, 'delta':delta}) #'delta_label':delta, 'avg_delta': avg_delta})
#                             else:
#                                 # Calculate the ideal library sizes and streak count
#                                 ideal_libsizes = np.arange(rel_df.LibSize.max(), surr_grp_df_pos.LibSize.min() - 1, -5)
#                                 ik = 0
#                                 cusp = 0
#                                 end = False
#                                 missing = []
#                                 for libsize in ideal_libsizes:
#                                     if libsize in surr_grp_df_pos.LibSize.values:
#                                         if not end:
#                                             ik += 1
#                                             cusp = libsize
#                                     else:
#                                         end = True
#                                         missing.append(libsize)
#
#                                 summary_d.update({'relation': relation, 'surr_var': surr_var, 'cusp': cusp, 'streak_count': ik, 'max_len': len(ideal_libsizes),'delta':delta})# 'delta_label':delta})
#                             libsize_summaries.append(summary_d)
#
#                     surr_run_pctile_list.append(surr_run_df_pctile)
#                 else:
#                     print(''
#                           '\tnot exists', ptiles_df_path, file=sys.stderr, flush=True)
#
#             if len(surr_run_pctile_list) > 0:
#                 # Consolidate percentile data frames
#                 surr_df_pctile_df_consol = pd.concat(surr_run_pctile_list)
#                 summary_info = []
#                 for relation, rel_df in surr_df_pctile_df_consol.groupby('relation'):
#                     matching_columns_delta = [col for col in rel_df if pattern.match(col)]
#
#                     for lib_size, _df in rel_df.groupby('LibSize'):
#                         num_real = rel_df.n_real.iloc[0]
#                         summary_d = summary_meta.copy()
#                         summary_d.update({'relation': relation, 'LibSize': lib_size, 'n_real': num_real, 'num_surr': len(surr_run_pctile_list), 'surr_var': surr_var})
#
#                         # Calculate averages of relevant percentiles
#                         matching_columns_surr = [col for col in _df if pattern_surr.match(col)]
#                         matching_columns_real = [col for col in _df if pattern_real.match(col)]
#                         for col in matching_columns_surr:
#                             summary_d[f'avg_{col}'] = _df[col].mean()
#                         for col in matching_columns_real:
#                             summary_d[f'avg_{col}'] = _df[col].mean()
#
#                         # summary_d['avg_real_5p'] = _df['r5p'].mean()
#                         # summary_d['avg_surr_95p'] = _df['s95p'].mean()
#                         # summary_d['avg_real_10p'] = _df['r10p'].mean()
#                         # summary_d['avg_surr_90p'] = _df['s90p'].mean()
#                         # summary_d['avg_real_25p'] = _df['r25p'].mean()
#                         # summary_d['avg_surr_75p'] = _df['s75p'].mean()
#
#                         # stat_cols = [col for col in _df.columns if col not in ['relation', 'LibSize', 'num_real']]
#                         # matching_columns = [col for col in stat_cols if pattern.match(col)]
#
#                         for col in matching_columns_delta:
#                             pct_pos = len(_df[_df[col] > 0]) / len(_df)
#                             summary_d[f'pct_plus_{col}'] = pct_pos
#                         summary_info.append(summary_d)
#
#                 if summary_info:
#                     summary_dfs.append(pd.DataFrame(summary_info))
#                     print('summary_info', len(summary_info), file=sys.stdout, flush=True)
#     return summary_dfs, libsize_summaries
#
# if __name__ == '__main__':
#     # Read the CSV file with calculation logs
#     calc_log2_df = pd.read_csv(calc_carc/'calc_log3.csv', index_col=0)
#
#     # Group by the specified columns
#     grouped = calc_log2_df.groupby(['temp_author', 'tsi_author', 'tau', 'E', 'knn', 'Tp', 'lag', 'sample', 'weighted', 'target_var', 'col_var'])
#
#     args = [(grp, grp_df) for grp, grp_df in grouped]
#
#     # Determine the number of CPUs to use from the SLURM environment variable
#     num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
#
#     # Use multiprocessing to parallelize the process
#     with Pool(num_cpus) as pool:
#         results = pool.map(process_group_summary, args)
#
#     summary_dfs = []
#     libsize_summaries = []
#
#     # Collect results from parallel processing
#     for summary_df, libsize_summary in results:
#         if summary_df:
#             summary_dfs.extend(summary_df)
#         if libsize_summary:
#             libsize_summaries.extend(libsize_summary)
#
#     # Combine the results into data frames
#     summary_df = pd.concat(summary_dfs) if summary_dfs else pd.DataFrame()
#     libsize_summaries_df = pd.DataFrame(libsize_summaries)
#
#     # Save the results to CSV files
#     summary_df.to_csv(calc_carc / 'summary_df.csv')
#     libsize_summaries_df.to_csv(calc_carc / 'libsize_summaries_df.csv')
