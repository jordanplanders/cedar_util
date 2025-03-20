
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import re
import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ast

from utils.arg_parser import get_parser, parse_flags, construct_convergence_name
from utils.config_parser import load_config
from utils.data_access import collect_raw_data, get_weighted_flag, check_for_exit, set_df_weighted, write_query_string
from utils.data_access import pull_percentile_data, get_group_sizes, get_sample_rep_n, check_empty_concat
from utils.data_processing import is_float

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.stats import t

from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

pd.option_context('mode.use_inf_as_na', True)

# pal = {'temp influences TSI': 'teal',
#        'temp influences TSI (surr)': 'dodgerblue',
#        'temp (surr) influences TSI': 'turquoise',
#
#        'TSI influences temp': 'red',
#        'TSI (surr) influences temp': 'indianred',
#        'TSI influences temp (surr)': 'darkorange'}

# def streamline_cause(label, split_word='causes'):
#     label_parts = label.split(' {} '.format(split_word))
#     clean_parts = []
#     for part in label_parts:
#         # print(part.split('(t-'))
#         try:
#             part = ' '.join(list(set([col.split(')')[-1].strip(' ') for col in part.split('(t-')]))).strip(' ')
#         except:
#             part = part.strip(' ')
#         clean_parts.append(part)
#     label = f' {split_word} '.join(clean_parts)
#     return label


#
# def calc_rs_delta(arg_tuple):
#     grp_d, convergence_d, config, delta, calc_convergence_dir, calc_location, real_dfs_references, function_flag, plot_flag, override, datetime_flag = arg_tuple
#     ci_level = 0.95
#
#     rel = config.target_relation
#
#     grp_d["group_id"] = int(grp_d["group_id"])
#     calc_grp_pctiles_parent = calc_location / config.calc_carc_dir.dirs.calc_grp_pctile_dir
#     calc_grp_pctiles_dir = calc_grp_pctiles_parent / str(int(grp_d['group_id']))
#
#     if grp_d['weighted'] in [True, 'True']:
#         weighted_flag = 'weighted'
#     else:
#         weighted_flag = 'unweighted'
#
#     # fig_sub_dir = raw_fig_dir/f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'
#     # fig_sub_dir.mkdir(exist_ok=True, parents=True)
#     # fig_target_path = fig_sub_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{delta}_{weighted_flag}_Tp{grp_d["Tp"]}_startind{grp_d["train_ind_i"]}_knn{grp_d["knn"]}_lag{grp_d["lag"]}_{grp_d["group_id"]}.png'
#
#
#     if convergence_d is None:#(override in [False, None]):
#         print('no convergence information found')
#         # print(f'figure: grp_id:{str(grp_d["group_id"])} already exists, skipping', file=sys.stdout, flush=True)
#         return
#     else:
#         start_time = datetime.datetime.now()
#         new_data = []
#         real_pctiles_path = calc_grp_pctiles_dir / f'{grp_d["group_id"]}_pctiles_real.csv'
#         if real_pctiles_path.exists() == True:
#             real_df_full = pd.read_csv(real_pctiles_path, index_col=0)
#             real_df_full['relation'] = real_df_full['relation'].apply(lambda x: streamline_cause(x))
#
#             real_df_min = real_df_full[(real_df_full['LibSize'] <=25 )].copy()
#             real_df_full = real_df_full[(real_df_full['LibSize'] >= convergence_d['convergence_libsize_min']) &
#                                         (real_df_full['LibSize'] <= convergence_d['convergence_libsize_max'])].copy()
#
#             delta_real = []
#
#             real_rho_min = real_df_min['mean'].values
#             real_rho_deltas = []
#             sample_num = real_df_full.groupby(['LibSize']).size().max()
#             for val in real_rho_min:
#                 for _ in range(sample_num):
#                     real_df_sample = real_df_full.groupby(['LibSize']).apply(
#                         lambda x: x.sample(n=1)).reset_index(drop=True)
#                     real_rho_deltas.append(real_df_sample['mean'].values - val)
#             real_rho_deltas = np.concatenate(real_rho_deltas)
#             res2 = stats.ttest_1samp(real_rho_deltas, popmean=0, nan_policy='omit')
#             ci = res2.confidence_interval(confidence_level=ci_level)
#             ci_low = ci.low
#             ci_high = ci.high
#             n = len(real_rho_deltas)
#             perc_pos = len(np.argwhere(real_rho_deltas > 0)) / n
#
#             grp_d_cp = grp_d.copy()
#             grp_d_cp['surr_var']= 'neither'
#             real_df_sample['relation'] = real_df_sample['relation'].apply(lambda x: streamline_cause(x))
#             grp_d_cp['relation'] = real_df_sample['relation'].unique()[0]
#             grp_d_cp.update({'condition':'delta_rho_pos', 'delta': delta, 'n': n, 'ci_low': ci_low, 'ci_high': ci_high, 'perc_pos': perc_pos})
#             new_data.append(grp_d_cp)
#
#         else:
#             print(f'no real pctiles found for {grp_d["group_id"]}')
#             return
#
#         # surrs_dfs = []
#         for surr_var in [config.target.var, config.col.var]:
#             surrs_dfs = []
#             surr_pctiles_path = calc_grp_pctiles_dir / f'{grp_d["group_id"]}_pctiles_{surr_var}2.csv'
#             if surr_pctiles_path.exists() == True:
#                 surr_var_df = pd.read_csv(surr_pctiles_path, index_col=0)
#                 surr_var_df = surr_var_df[(surr_var_df['LibSize'] >= convergence_d['convergence_libsize_min']) &
#                                             (surr_var_df['LibSize'] <= convergence_d['convergence_libsize_max'])].copy()
#                 surrs_dfs.append(surr_var_df)
#
#             surr_df_full = pd.concat(surrs_dfs)
#             print('psets', surr_df_full.pset_id.unique())
#             surr_df_full_lst = [df.drop_duplicates(subset=['s5p'], ignore_index=True) for _, df in
#                                 surr_df_full.groupby('pset_id')]
#
#
#         delta_dfs = []
#         if len(surr_df_full) > 0:
#             for _ in range(50):
#                 real_df_sample = real_df_full.groupby(['LibSize']).apply(
#                     lambda x: x.sample(n=1)).reset_index(drop=True)
#
#                 real_df_sample['relation'] = real_df_sample['relation'].apply(lambda x: streamline_cause(x))
#
#                 sampling = np.arange(0, len(surr_df_full_lst))#np.random.randint(0, len(surr_df_full_lst) - 1, size=int(.2 * len(surr_df_full_lst)))
#                 for iq in sampling:
#                     _tmp = real_df_sample.copy()
#                     _tmp = _tmp.rename(columns={'mean': 'mean_r'})
#                     _tmp_surr = surr_df_full_lst[iq].copy()
#
#                     _tmp_surr= _tmp_surr[['LibSize', 'relation_r', 'relation_s', 's50p', 'mean_s', 'pset_id']].copy()
#                     _tmp = _tmp_surr.merge(_tmp, right_on=['LibSize', 'relation'], left_on=['LibSize', 'relation_r'],
#                                       suffixes=('_r', '_s'))
#
#                     if delta == 'rmean-smean':
#                         _tmp[delta] = _tmp['mean_r']-_tmp['mean_s']
#                     elif delta == 'r50p-s50p':
#                         _tmp[delta] = _tmp['s50p']-_tmp['s50p']
#
#                     delta_dfs.append(_tmp)
#
#             delta_df_full = pd.concat(delta_dfs)
#             if len(delta_df_full) == 0:
#                 print('no delta_df_full', grp_d)
#                 return
#
#             for relation_s, rel_s_df in delta_df_full.groupby('relation_s'):
#
#                 res2 = stats.ttest_1samp(rel_s_df[delta], popmean=0, nan_policy='omit')
#                 ci = res2.confidence_interval(confidence_level=ci_level)
#                 ci_low = ci.low
#                 ci_high = ci.high
#                 n = len(rel_s_df)
#                 perc_pos = len(rel_s_df[rel_s_df[delta] > 0]) / n
#
#                 grp_d_cp = grp_d.copy()
#                 grp_d_cp['surr_var'] = surr_var
#                 grp_d_cp['relation'] = relation_s
#
#                 grp_d_cp.update({'condition': 'delta_rs', 'delta': delta, 'n': n, 'ci_low': ci_low, 'ci_high': ci_high, 'perc_pos': perc_pos})
#                 new_data.append(grp_d_cp)
#         # print('in situ calc', 'len(delta_df)',len(delta_df_full), 'len(delta_dfs)', [len(df) for df in delta_dfs][:10])
#
#         # return dictionary with group information, surr_var, percent delta >0 over convergence interval, n, delta_label
#         return pd.DataFrame(new_data)

#
# def calc_slope_metrics(slope_df, tolerance=0.05):
#     slope_d = slope_df[['LibSize', 'window', 'relation', 'start_rho']].drop_duplicates().to_dict('records')[0]
#     # Compare to a zero-mean distribution with similar variance
#     # res_conv = stats.ttest_1samp(slopes, 0)
#     # test_dist = np.random.normal(0, 1E-4, len(slopes))
#     res_conv = stats.ttest_1samp(slope_df['slope'], 0)  #, equal_var=False)
#
#     # a = -1E-4
#     # b = 1E-4
#
#     # for min-max scaled rho values
#     a = -.01  #*slope_d['start_rho']/200#-delta_rho/100#.00125*delta_rho#-1E-4
#     b = .01  #*slope_d['start_rho']/200#delta_rho/100#.00125*delta_rho#1E-4
#
#     # Count the number of elements within the range [a, b]
#     # Randomly sample 5 elements without replacement
#     n = len(slope_df)  #min([len(slopes), n])
#     # sample = np.random.choice(slopes, size=n, replace=False)
#     m = np.sum((slope_df['perc_change'].values >= a) & (slope_df['perc_change'].values <= b))
#     # Calculate the percentage
#     pct_conv = (m / n)
#
#     # print(f"Percentage of elements in the range [{a}, {b}]: {percentage}, {m}")
#     m = np.sum((slope_df['perc_change'].values > b))
#     pct_inc = (m / n)
#
#     m = np.sum((slope_df['perc_change'].values < a))
#     pct_dec = (m / n)
#
#     ci_level = 1 - tolerance
#     ci = res_conv.confidence_interval(confidence_level=ci_level)
#     ci_low = ci.low  #min(a, ci.low)
#     ci_high = ci.high  #max(b, ci.high)
#     mean_slope = slope_df['slope'].mean()
#
#     slope_d.update({'mean': mean_slope, 'ci_mean_low': ci_low, 'ci_mean_high': ci_high, 'pvalue': res_conv.pvalue,
#                     'pct_conv': pct_conv, 'pct_dec': pct_dec, 'pct_inc': pct_inc})
#     return slope_d

#
# def plot_max_libsize_vs_metric_with_error(arg_tuple):
#     grp_d, config, calc_convergence_dir, real_dfs_references, plot_flag, override_flag, datetime_flag = arg_tuple
#
#     start_time = datetime.datetime.now()
#     print(f'grp_id:{str(grp_d["group_id"])} received', start_time, file=sys.stdout, flush=True)
#     pal = config.pal.to_dict()
#     calc_metric_csvs = calc_convergence_dir / config.calc_convergence_dir.csvs  # 'csvs2'
#     metric_df_path = calc_metric_csvs / f'{str(grp_d["group_id"])}.csv'
#     fig_dir = calc_convergence_dir / config.calc_convergence_dir.figures  # calc_metric_dir / 'figures2'
#
#     if grp_d['weighted'] in ['True', True]:
#         weighted_flag = 'weighted'
#     elif grp_d['weighted'] in ['False', False]:
#         weighted_flag = 'unweighted'
#     # Read in the CSV
#     try:
#         df = pd.read_csv(metric_df_path)
#     except:
#         return
#
#     for metric in ['L', 'k']:
#         # Parse the confidence intervals from the CSV
#         df[f'{metric}_CI_lower'] = df[f'{metric}_CI'].apply(lambda x: float(x.strip('()').split(',')[0]))
#         df[f'{metric}_CI_upper'] = df[f'{metric}_CI'].apply(lambda x: float(x.strip('()').split(',')[1]))
#
#         # Calculate error bars
#         df[f'{metric}_error_lower'] = df[metric] - df[f'{metric}_CI_lower']
#         df[f'{metric}_error_upper'] = df[f'{metric}_CI_upper'] - df[metric]
#
#         # Create a plot using Seaborn
#         fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#         ax2 = ax.twinx()
#
#         # Create the plot with error bars using seaborn
#         ax = sns.scatterplot(x='max_lib_size', y=metric, data=df, hue='relation', palette=pal, ax=ax)
#         ax2 = sns.scatterplot(x='max_lib_size', y=f'{metric}_error', data=df, hue='relation', palette=pal, ax=ax2)
#
#         # Add error bars using matplotlib
#         for rel, rel_df in df.groupby('relation'):
#             plt.errorbar(rel_df['max_lib_size'], rel_df[metric], yerr=[rel_df[f'{metric}_error_lower'],
#                                                                        rel_df[f'{metric}_error_upper']], fmt='o',
#                          color=pal[rel],
#                          ecolor=pal[rel], elinewidth=2, capsize=4)
#
#         # Add labels and title
#         ax.set_xlabel('Max Library Size')
#         ax.set_ylabel(f'{metric}')
#
#         ax2.set_ylabel(f'{metric} error')
#
#         # plt.title('Max Library Size vs L with Confidence Intervals')
#
#         fig_name = f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}_{metric}_{str(grp_d["group_id"])}.png'
#
#         # fig_dir = calc_metric_dir / config['calc_criteria_rates2']['calc_metrics_dir']['figures']
#
#         # fig_dir = calc_metric_dir / 'figures2'
#         spec_fig_dir = fig_dir / metric
#         # target_relation.replace(' ',
#         #                                  '_') / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_slim')
#         if spec_fig_dir.exists() is False:
#             spec_fig_dir.mkdir(exist_ok=True, parents=True)
#
#         fig.savefig(spec_fig_dir / fig_name, dpi=300, bbox_inches='tight')
#         plt.close()
#
#

#
# def binding_func(x, bmax, x0, kd, rho_0):
#     # Ensure that L is always greater than or equal to L_0
#     L_adjusted = np.maximum(x - x0, 0)
#     return rho_0 + (bmax - rho_0) * (L_adjusted) / (kd + L_adjusted)

    # return ((bmax) * (x-x0)) / ((x-x0) + kd)
#
# def inverse_exponential_func(x, L, x_initial, c, rho_initial):
#     return (L * rho_initial * np.exp(c * (x - x_initial))) / (
#             L + rho_initial * np.exp(c * (x - x_initial)) - rho_initial)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
#
# def compute_convergence_point(libsizes, params, binding_func):
#     # Calculate the binding curve values
#     rho_values = binding_func(libsizes, **params)
#
#     # Compute the gradient (slope)
#     slopes = np.gradient(rho_values, libsizes)
#
#     # Find the point where the slope is smallest (i.e., closest to zero)
#     norm_slopes = np.abs(slopes) / np.max(np.abs(slopes))
#     convergence_idx = np.argmin(np.abs(norm_slopes))
#     flag = False
#     if norm_slopes[convergence_idx] <= .005:
#         flag = True
#     convergence_slope = slopes[convergence_idx]
#     convergence_libsize = libsizes[convergence_idx]
#     convergence_rho = rho_values[convergence_idx]
#
#     return convergence_libsize, convergence_rho, convergence_slope, slopes, rho_values, flag
#

# def compute_convergence_intervals(libsizes, params, binding_func):
#     # Calculate the binding curve values
#     rho_values = binding_func(libsizes, **params)
#
#     # Compute the gradient (slope)
#     slopes = np.gradient(rho_values, libsizes)
#
#     # Find the point where the slope is smallest (i.e., closest to zero)
#     norm_slopes = np.abs(slopes) / np.max(np.abs(slopes))
#     try:
#         convergence_idx = np.argwhere(norm_slopes<=.005).flatten()#np.argmin(np.abs(slopes))
#         splits = np.argwhere(np.diff(convergence_idx)>1).flatten()
#         if len(splits)>0:
#             convergence_idx = convergence_idx[convergence_idx>splits[0]][0]
#         convergence_norm_slopes = norm_slopes[convergence_idx].flatten()
#         convergence_libsizes = libsizes[convergence_idx].flatten()
#         convergence_rhos = rho_values[convergence_idx].flatten()
#     except:
#         convergence_idx = []
#         convergence_norm_slopes = []
#         convergence_libsizes = []
#         convergence_rhos = []
#
#     return convergence_libsizes, convergence_rhos, convergence_norm_slopes
#
#     # First, make sure we find the correct convergence location (filter out false initial flat spot)
#     # Calculate the differences between consecutive libsizes
#

# def find_relevant_streaks(convergence_libsizes, convergence_rhos):
#     differences = np.diff(convergence_libsizes)
#     # print('differences', differences)
#     # Find the break points where the difference is not 1
#     split_points = np.argwhere(differences > 5)  # [0] + 1
#
#     if len(split_points) == 0:
#         # if len(np.argwhere(differences <= 15)) == len(differences):
#         split_points = np.array([len(convergence_libsizes)])
#     # else:
#     #     print('convergence_libsizes', convergence_libsizes)
#     # split_points = [len(convergence_libsizes)]
#     # print('split_points', split_points)
#     else:
#         split_points = np.append(split_points, len(convergence_libsizes))
#
#     # print('split_points2', split_points)
#     # print('convergence_libsizes2', convergence_libsizes)
#     # Split the indexes array at these break points
#     streaks = np.split(convergence_libsizes, split_points)
#     streak_rhos = np.split(convergence_rhos, split_points)
#
#     # print('streaks', streaks)
#     if len(streaks) > 1:
#         if streaks[0][0] < 30:
#             streaks = streaks[1:]
#             streak_rhos = streak_rhos[1:]
#
#     return streaks, streak_rhos

#
# def check_overlap(real, convs):
#     # real_rho, real_error = real
#     real_df_rel_rho_l, real_df_rel_rho_u = real  # - real_error, real + real_error
#     # print(real_df_rel_rho_l, real_df_rel_rho_u)
#     conv_rho, conv_error = convs
#     conv_l = conv_rho - conv_error
#     conv_u = conv_rho + conv_error
#     # print(conv_l, conv_u, type(conv_l), type(conv_u))
#     if (((real_df_rel_rho_l >= conv_l) & (real_df_rel_rho_l <= conv_u)) |
#             ((real_df_rel_rho_u >= conv_l) & (real_df_rel_rho_u <= conv_u))):
#         # (
#         #         real_rho >= conv_l and real_rho <= conv_u):
#         return True
#     else:
#         return False
from matplotlib.patches import Rectangle
#
# def get_u_l(m, percent_threshold=.01):
#     m_l = m - np.abs(m) * percent_threshold
#     m_u = m + np.abs(m) * percent_threshold
#     return m_l, m_u

#
# def check_overlap2(real_rho, pred_rho, percent_threshold):
#     lower_bound_pred = pred_rho * (1 - percent_threshold)
#     upper_bound_pred = pred_rho * (1 + percent_threshold)
#
#     lower_bound_real = real_rho * (1 - percent_threshold)
#     upper_bound_real = real_rho * (1 + percent_threshold)
#
#     if lower_bound_pred <= upper_bound_real and lower_bound_pred >= lower_bound_real:
#         return True
#     if upper_bound_pred >= lower_bound_real and upper_bound_pred <= upper_bound_real:
#         return True
#     return False


#
# def calculate_fit(x, y_actual, row, model):
#     """
#     Calculates the goodness-of-fit for a given row dictionary of parameters.
#
#     Args:
#         x: The independent variable (LibSize in your case).
#         y_actual: The actual rho values.
#         row: A dictionary containing the parameters for the binding function.
#
#     Returns:
#         mse: Mean Squared Error between actual and predicted values.
#         r2: R-squared value indicating the goodness-of-fit.
#     """
#     y_pred = model(x, **row)
#     mse = np.sqrt(mean_squared_error(y_actual, y_pred))
#     mae = mean_absolute_error(y_actual, y_pred)
#     mape = np.mean(np.abs((y_actual - y_pred) / y_actual))
#     # r2 = r2_score(y_actual, y_pred)
#     return mse, mae, mape

#
# def test_statistical(libsize_df, early_threshold, model, func_config_d):
#     early_data = libsize_df[libsize_df['LibSize'] <= early_threshold].copy()
#     rhos = early_data['rho'].values
#
#
#
#     # Nonlinear model prediction
#     y_pred_nonlinear = model(early_data['LibSize'], **func_config_d)
#
#     # Constant model prediction (mean of rho)
#     rho_ref = y_pred_nonlinear.values[-1]#np.mean(rhos[-int(.2 * len(rhos)):])  # func_config_d['L']# =
#     y_pred_constant = np.full_like(early_data['rho'], rho_ref)
#
#     # Calculate errors
#     mae_constant = np.around(mean_absolute_error(early_data['rho'], y_pred_constant), 2)
#     mae_nonlinear = np.around(mean_absolute_error(early_data['rho'], y_pred_nonlinear), 2)
#     mse_constant = mean_squared_error(early_data['rho'], y_pred_constant)
#     mse_nonlinear = mean_squared_error(early_data['rho'], y_pred_nonlinear)
#
#     if mae_constant <=  mae_nonlinear:
#         status= 'statistical'
#         print(status)
#     else:
#         status = 'deterministic'
#
#     return status, mae_constant, mae_nonlinear

#
# def weighted_mae(y_actual, y_pred, weights):
#     return np.sum(weights * np.abs(y_actual - y_pred)) / np.sum(weights)

#
# def identify_streaks(libsize_sub, func_config_d, reg_func, test_interval_start, spacing, percent_threshold, fine_spacing=None):
#     # Implement streak logic
#     streak_count = 0
#     streak_data = dict(libsize=[], rho=[], rho_pred=[])
#     max_streak_data = dict(libsize=[], rho=[], rho_pred=[])
#     tight_streak_data = dict(libsize=[], rho=[], rho_pred=[])
#     max_streak_fine_data = dict(libsize=[], rho=[], rho_pred=[])
#     max_streak = 0
#     gap_count = 0
#
#     if fine_spacing is None:
#         fine_spacing = int(spacing*.75)
#
#     # assess each libsize bin
#     for libsize, libsize_df in libsize_sub[libsize_sub['LibSize'] >= test_interval_start].groupby('LibSize'):
#         pred_rho = reg_func(libsize, **func_config_d)
#         real_rho = libsize_df['rho'].median()
#
#         if len(streak_data['libsize']) > 0:
#             dt = libsize - streak_data['libsize'][-1]
#         else:
#             dt = 0
#
#         if dt > spacing:
#             streak_count = 0
#             streak_data = dict(libsize=[], rho=[], rho_pred=[])
#             tight_streak_data = dict(libsize=[], rho=[], rho_pred=[])
#
#         if dt > fine_spacing:
#             tight_streak_data = dict(libsize=[], rho=[], rho_pred=[])
#
#         if check_overlap2(real_rho, pred_rho, percent_threshold) is True:
#             # if libsize >= test_interval_start:
#             streak_count += 1
#             streak_data['libsize'].append(libsize)
#             streak_data['rho'].append(real_rho)
#             streak_data['rho_pred'].append(pred_rho)
#             gap_count = 0
#
#             tight_streak_data['libsize'].append(libsize)
#             tight_streak_data['rho'].append(real_rho)
#             tight_streak_data['rho_pred'].append(pred_rho)
#
#
#         if streak_count > max_streak:
#             max_streak = streak_count
#             max_streak_data = streak_data.copy()
#             max_streak_fine_data = tight_streak_data.copy()
#
#     # func_config_d['max_streak'] = max_streak
#
#     return max_streak_data, max_streak_fine_data, max_streak

#
# def plot_max_libsize_vs_rho_wmetric_with_error(arg_tuple):
#
#     def inverse_exponential(x, **kwargs):
#         L = kwargs['L']
#         x0 = kwargs['x0']
#         k = kwargs['k']
#         off = kwargs['off']
#         return inverse_exponential_func(x, L, x0, k, off)
#
#     def binding(x, **kwargs):
#         Kd = kwargs['Kd']
#         L = kwargs['L']
#         x0 = kwargs['x0']
#         off = kwargs['off']
#         return binding_func(x, L, x0, Kd, off)
#
#
#     (grp_d, ind, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
#      function_flag, plot_flag, override_flag, write_flag, datetime_flag, percent_threshold) = arg_tuple
#
#     meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
#        'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
#        'col_var', 'surr_var', 'col_var_id', 'target_var_id']
#
#     convergence_grps = []
#     _max_libsize=350
#     min_libsize_threshold = int(.33 * _max_libsize)
#     streak_threshold = 20
#     libsize_threshold = .5* _max_libsize
#     max_gap_allowed = 12
#     Kd_threshold = 1.5
#     metric='L'
#     knn=20
#     surr_testing_window=75
#
#
#     start_time = datetime.datetime.now()
#     print(f'grp_id:{str(grp_d["group_id"])} received', start_time, file=sys.stdout, flush=True)
#     pal = config.pal.to_dict()
#     calc_metric_csvs = calc_convergence_dir /config.calc_convergence_dir.dirs.csvs  # 'csvs2'
#     metric_df_path = calc_metric_csvs / f'{str(grp_d["group_id"])}.csv'
#     fig_dir = convergence_dir_csv_parts.parent /config.calc_convergence_dir.dirs.figures  # calc_metric_dir / 'figures2'
#     spec_fig_dir = fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'Tp{grp_d["Tp"]}'  #/f'lag{grp_d["lag"]}'
#     spec_fig_dir.mkdir(exist_ok=True, parents=True)
#
#
#     weighted_flag = get_weighted_flag(grp_d)
#     if function_flag == 'inverse_exponential':
#         reg_func = inverse_exponential
#     elif function_flag == 'binding':
#         reg_func = binding
#
#     try:
#         _df_func_configs = pd.read_csv(metric_df_path)
#     except:
#         print('no metric_df_path', metric_df_path, file=sys.stderr, flush=True)
#         return
#
#     fig_name = f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}__libsize_rho_{metric}__{str(grp_d["group_id"])}.png'
#
#     if override_flag in [False, 'False']:
#         if write_flag in [False, 'False']:
#             if (spec_fig_dir / fig_name).exists():
#                 print(f'figure: grp_id:{str(grp_d["group_id"])} already exists, skipping', file=sys.stdout, flush=True)
#                 return
#
#     meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
#                       'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
#                       'col_var', 'surr_var', 'col_var_id', 'target_var_id']
#     # collect data
#     real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)
#     real_df_full = real_df_full[real_df_full['LibSize'] >= knn].copy()
#     real_df_full= real_df_full[real_df_full['LibSize']<=_max_libsize].copy()
#     if len(real_df_full) == 0:
#         print('no real data found', grp_d, file=sys.stderr, flush=True)
#         return
#
#     pctile_range = [.25, .75]
#     sample_kwargs = {'n': 40, 'replace': False}
#     iqr_dfs = []
#     filter_var = 'rho'
#     groupby_var = 'LibSize'
#     for rel, real_df_full_rel in real_df_full.groupby('relation'):
#         if len(real_df_full_rel)==0:
#             continue
#
#         samples_df_list, samples_flags = bootstrap_raw_output(real_df_full_rel, filter_var, groupby_var, pctile_range, sample_kwargs=sample_kwargs, grp_d=grp_d)
#         iqr_dfs.extend(samples_df_list)
#
#     real_df = check_empty_concat(iqr_dfs)
#     if len(real_df)==0:
#         print('no real data found', grp_d, file=sys.stderr, flush=True)
#         return
#
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#     real_df = real_df[~real_df['relation'].str.isnumeric()].copy()
#     sns.lineplot(data=real_df, x='LibSize', y='rho', ax=ax, hue='relation', palette=pal, legend=False,
#                  alpha=.5, errorbar=('pi', 50))
#     sns.lineplot(data=real_df, x='LibSize', y='rho', ax=ax, hue='relation', palette=pal, legend=False,
#                  alpha=.85, errorbar=('pi', 1))
#
#
#     # Compare models
#     # for each model:
#     cancel = False
#     for rel, df_func_configs in _df_func_configs.groupby('relation'):
#         df_func_configs =df_func_configs.sort_values(by='max_lib_size', ascending=True)
#         # choose last value, this could be aggregate and take mean also
#         df_func_configs = df_func_configs.groupby('max_lib_size').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
#         status = []
#         real_df_rel = real_df[real_df['relation'] == rel].copy()
#
#
#         ## Test for statistical relationship
#         # mean_Kd = df_func_configs[df_func_configs['max_lib_size']<=min_libsize_threshold]['Kd'].mean()
#         # Kd_status = '(Kd = {:.2f})'.format(mean_Kd)
#
#         # approach 1: is the Kd for small libsizes too low?
#         # if mean_Kd<=Kd_threshold:
#         #     # cancel=False
#         #     status.append('Kd is low')
#         #     sns.scatterplot(data=df_func_configs, x='max_lib_size', y=metric, ax=ax, color='teal',
#         #                     legend=False,
#         #                     **{'s': 5, 'edgecolor': 'k', 'linewidth': .5,'zorder':100})
#         # else:
#         #     Kd_status = 'Kd ok '+Kd_status
#
#         # approach 2: for libsizes smaller than a threshold, is a flat line more predictive than the binding curve?
#
#         func_config_d = df_func_configs[df_func_configs['max_lib_size'] > min_libsize_threshold].iloc[0].to_dict().copy()
#
#         try:
#             stat_status, mae_constant, mae_nonlinear = test_statistical(real_df_rel, min_libsize_threshold, reg_func, func_config_d)
#             stat_detail = f' (mae_const: {mae_constant:.2f}, mae_nonlin: {mae_nonlinear:.2f})'
#             if stat_status == 'statistical':
#                 # cancel = True
#                 # status.append(stat_status)
#                 sns.scatterplot(data=df_func_configs, x='max_lib_size', y=metric, ax=ax, color='yellow',
#                                 legend=False,
#                                 **{'s': 5, 'edgecolor':'k','linewidth': .5})
#             status.append(stat_status+stat_detail)
#         except:
#             print('exception at stat test', grp_d, file=sys.stderr, flush=True)
#
#         if cancel is True:
#             _tmp_libsizes = real_df_rel[(real_df_rel['LibSize']<=min_libsize_threshold)]['LibSize'].unique()
#             _tmp_libsizes.sort()
#             try:
#
#                 convergence_libsizes, convergence_rhos, convergence_norm_slopes = compute_convergence_intervals(
#                     _tmp_libsizes, func_config_d, reg_func)
#                 convergence_libsize, convergence_rho, convergence_slope, slopes, rho_values, convergence_flag = compute_convergence_point(
#                     _tmp_libsizes, func_config_d, reg_func)
#                 norm_slopes = np.abs(slopes) / np.max(np.abs(slopes))
#
#                 norm = Normalize(vmin=0, vmax=.05)
#                 sm = ScalarMappable(cmap="Greys", norm=norm)
#                 for i in range(len(_tmp_libsizes) - 1):
#                     ax.plot(_tmp_libsizes[i:i + 2], reg_func(_tmp_libsizes[i:i + 2], **func_config_d),
#                             color=sm.to_rgba(norm_slopes[i]), alpha=1, linewidth=1, zorder=1)
#
#                 if convergence_flag is True:
#                     ax.scatter([convergence_libsizes[0]], [convergence_rhos[0]], color='b', s=10, zorder=100)
#                 else:
#                     ax.scatter([convergence_libsize], [convergence_rho], color='g', s=10, zorder=100,marker='o' )
#                 # ax.scatter([convergence_libsize], [convergence_rho], color='b', s=10, zorder=100)
#
#             except:
#                 print('tried to plot to explain cancel', func_config_d, file=sys.stderr, flush=True)
#
#         # if cancel is False:
#         func_evals = []
#
#         # process each function fit configuration
#         max_streak_data_lst = []
#         # @todo need to groupby max_lib_size and aggregate over config parameters and use aggregate for this
#         # df_func_configs = df_func_configs.groupby(['max_lib_size'])
#
#         for idx, _func_config_d in df_func_configs.iterrows():
#
#             func_config_d = _func_config_d.to_dict().copy()
#             config_max_libsize = func_config_d['max_lib_size']
#             streak_distance_threshold = .3 * config_max_libsize
#             Kd = func_config_d['Kd']
#
#             # establish hypothetical convergence point
#             _tmp_libsizes = np.arange(real_df_rel["LibSize"].min(), 1200, 5)  # row['max_lib_size'], 5)
#             _tmp_rhos = reg_func(_tmp_libsizes, **func_config_d)
#
#             convergence_libsizes_ideal, convergence_rhos_ideal, convergence_norm_slopes = compute_convergence_intervals(
#                 _tmp_libsizes,func_config_d,reg_func)
#
#             if len(convergence_libsizes_ideal)==0:
#                 ideal_exists= False
#                 print('no convergence found', func_config_d, file=sys.stderr, flush=True)
#
#                 convergence_libsize_ideal = _max_libsize
#                 convergence_rho_ideal = reg_func(np.array([_max_libsize]), **func_config_d)[0]
#             else:
#                 ideal_exists= True
#                 convergence_libsize_ideal = convergence_libsizes_ideal[0]
#                 convergence_rho_ideal = convergence_rhos_ideal[0]
#             #
#             # start_ind = np.argwhere(_tmp_libsizes>max(min(_tmp_libsizes)+5, convergence_libsize))[0][0]
#             # convergence_libsizes = _tmp_libsizes[start_ind:]
#             # convergence_rhos = _tmp_rhos[start_ind:]
#             #
#             # convergence_libsize_ideal = convergence_libsizes_ideal[0]
#             # convergence_rho_ideal = convergence_rhos_ideal[0]
#
#             # ax.scatter([convergence_libsize_ideal], [convergence_rho_ideal], color='b', s=10, zorder=100, marker='o')
#
#             ### establish end test interval
#             # set interval bounds
#             test_interval_width = int(min(streak_threshold*1.25*5, .5*config_max_libsize)) #streak_distance_threshold
#             # if the convergence libsize is within range, set the interval to follow the convergence libsize
#             test_interval_start = Kd*6+(knn+5)
#             test_interval_end = _max_libsize
#
#             if test_interval_start >= _max_libsize-streak_threshold*1.25*5:
#                 test_interval_start= _max_libsize-streak_threshold*1.25*5
#             # if convergence_libsize_ideal < (_max_libsize-test_interval_width):#func_config_d['max_lib_size'] - test_interval_width:
#             #     test_interval_start = convergence_libsize_ideal-int(.5*test_interval_width) #min(150, convergence_libsize_ideal)
#             #     test_interval_end = _max_libsize#test_interval_start + test_interval_width
#             # # otherwise, specify an interval at the end of the range
#             # else:
#             #     test_interval_start = min(150, _max_libsize-test_interval_width)#func_config_d['max_lib_size'] - test_interval_width - 1
#             #     test_interval_end = _max_libsize#func_config_d['max_lib_size']
#
#
#             ## get down to the comparison
#             # already filtered for IQR
#             libsize_sub =real_df_rel[real_df_rel["LibSize"] <= test_interval_end].copy()#func_config_d['max_lib_size']+streak_threshold*5].copy()
#
#             # Example weighting: more weight to early libsizes
#             libsizes = libsize_sub['LibSize'].values
#             max_libsize = libsizes.max()
#             weights = libsizes / _max_libsize#max_libsize  # Weights decrease with increasing libsize
#
#             y_pred = reg_func(libsize_sub['LibSize'], **func_config_d)
#             libsize_sub['pred_rho'] = y_pred
#             # Calculate the weighted MAE
#             w_mae = weighted_mae(libsize_sub['rho'], y_pred, weights)
#             func_config_d['full_wmae'] = w_mae
#
#             # rough cut
#             spacing = 21
#             max_streak_data, max_streak_fine_data, max_streak = identify_streaks(libsize_sub, func_config_d, reg_func,
#                                                                                  test_interval_start, spacing, percent_threshold)
#
#             func_config_d['max_streak'] = max_streak
#             if len(max_streak_fine_data['libsize'])>0:
#                 np.argmin(max_streak_fine_data['libsize'])
#                 max_streak_data['surr_libsize_min'] = min(max_streak_fine_data['libsize'])
#                 max_streak_data['surr_rho_max'] = np.mean(max_streak_fine_data['rho'])
#                 max_streak_data['surr_rho_min'] = np.mean(max_streak_fine_data['rho'])
#                 if max( max_streak_fine_data['libsize'])-min(max_streak_fine_data['libsize'])>=50:
#                     max_streak_data['surr_libsize_max'] = max(max_streak_fine_data['libsize'])
#
#                 else:
#                     max_streak_data['surr_libsize_max'] = min(max_streak_fine_data['libsize'])+50
#             elif len(max_streak_data)>0:
#                 max_streak_data['surr_libsize_min'] = min(max_streak_data['libsize'])
#                 max_streak_data['surr_libsize_max'] = min(max_streak_data['libsize'])+50
#                 max_streak_data['surr_rho_max'] = np.mean(max_streak_data['rho'])
#                 max_streak_data['surr_rho_min'] = np.mean(max_streak_data['rho'])
#
#                 # max_streak_fine_data = max_streak_data
#             max_streak_data_lst.append(max_streak_data)
#
#             func_evals.append(func_config_d)
#
#         if len(func_evals)==0:
#             continue
#
#         eval_df = pd.DataFrame(func_evals)#.to_csv(spec_fig_dir / f'{rel}_func_evals.csv', index=False)
#         passing_models = eval_df[eval_df['max_streak'] >= streak_threshold]
#
#         if not passing_models.empty:
#             # Select the model with the lowest RMSE among the passing models
#             best_model_idx = passing_models['full_wmae'].idxmin()
#             max_streak_data_d = max_streak_data_lst[best_model_idx]
#             best_model = passing_models.loc[best_model_idx]
#             best_row = best_model.to_dict()
#             status = ['Pass']+status
#             color = 'c'
#             details_d = best_row.copy()
#
#             max_streak_data_d['libsize'] = np.array(max_streak_data_d['libsize'])
#             sort_inds = np.argsort(max_streak_data_d['libsize'])
#             max_streak_data_d['rho'] = np.array(max_streak_data_d['rho'])[sort_inds]
#
#
#             alignment_libsize_min = min(max_streak_data_d['libsize'])
#             alignment_libsize_max = max(max_streak_data_d['libsize'])
#
#             if convergence_libsize_ideal < alignment_libsize_min:
#                 convergence_libsize_ideal = alignment_libsize_min
#             elif convergence_libsize_ideal < alignment_libsize_max:
#                 alignment_libsize_max = min(_max_libsize, min(convergence_libsize_ideal+ surr_testing_window, alignment_libsize_max))
#
#             if convergence_libsize_ideal > _max_libsize:
#                 convergence_libsize_ideal = _max_libsize
#
#             alignment_libsizes = real_df_rel[(real_df_rel['LibSize'] <= alignment_libsize_max-surr_testing_window) & (real_df_rel['LibSize'] >= alignment_libsize_min)]['LibSize'].unique()
#             alignment_libsizes = alignment_libsizes[alignment_libsizes<=_max_libsize]
#             if len(alignment_libsizes)==0:
#                 alignment_libsizes = np.array([alignment_libsize_min])
#             else:
#                 alignment_libsizes.sort()
#
#             window_stats = []
#             for interval_start in alignment_libsizes:
#                 window = real_df_rel[(real_df_rel['LibSize'] <= interval_start+surr_testing_window) & (real_df_rel['LibSize'] >= interval_start)]
#                 if len(window) == 0:
#                     continue
#
#                 window_median = window['rho'].median()
#                 window_rhos = window['rho'].values
#                 # window_pred = reg_func(i, **best_row)
#                 window_mae = np.mean(np.abs(window_median - window_rhos))
#                 window_stats.append({'interval_start': interval_start, 'mae': window_mae, 'window_median': window_median, 'interval_end': interval_start+surr_testing_window})
#
#
#             window_stats_df = pd.DataFrame(window_stats)
#             window_stats_df = window_stats_df.sort_values(by='mae', ascending=True)
#             convergence_libsize_min = window_stats_df.iloc[0]['interval_start']
#             convergence_libsize_max = min(window_stats_df.iloc[0]['interval_end'], _max_libsize)
#             convergence_rho_min = window_stats_df.iloc[0]['window_median']
#             convergence_rho_max = window_stats_df.iloc[0]['window_median']
#
#             details_d.update({
#                 'convergence_libsize_min':convergence_libsize_min,
#                 'convergence_libsize_max':convergence_libsize_max,
#                 'convergence_rho_min':convergence_rho_min,
#                 'convergence_rho_max':convergence_rho_max,
#                 'surr_libsize_min':max_streak_data_d['surr_libsize_min'],
#                 'surr_libsize_max':max_streak_data_d['surr_libsize_min']+surr_testing_window,
#                 'surr_rho_min':max_streak_data_d['surr_rho_min'],
#                 'surr_rho_max':max_streak_data_d['surr_rho_min']})
#             print({
#                 'convergence_libsize_min':convergence_libsize_min,
#                 'convergence_libsize_max':convergence_libsize_max,
#                 'convergence_rho_min':convergence_rho_min,
#                 'convergence_rho_max':convergence_rho_max})
#
#
#             window = real_df_rel[real_df_rel['LibSize'] <= convergence_libsize_max+1 ]
#             _tmp_libsizes = window['LibSize'].unique()
#             _tmp_libsizes.sort()
#
#             libsize_width =max(_tmp_libsizes)-convergence_libsize_min #max(passing_libsizes) - min(passing_libsizes)
#             rho_height = 2 * convergence_rho_min * percent_threshold
#             ax.add_patch(
#                 Rectangle(xy=(convergence_libsize_min, convergence_rho_min - .5 * rho_height),
#                           width=libsize_width,
#                           height=rho_height, facecolor=None, alpha=.3, zorder=500, edgecolor=color)
#             )
#
#             libsize_width = max_streak_data_d['surr_rho_max']-max_streak_data_d['surr_rho_min'] #max(passing_libsizes) - min(passing_libsizes)
#             rho_height = 2 * max_streak_data_d['surr_rho_min'] * percent_threshold
#             ax.add_patch(
#                 Rectangle(xy=(max_streak_data_d['surr_rho_min'], max_streak_data_d['surr_rho_min'] - .5 * rho_height),
#                           width=libsize_width,
#                           height=rho_height, facecolor="none", alpha=.5, zorder=500, edgecolor=color)
#             )
#
#             convergence_libsize, convergence_rho, convergence_slope, slopes, rho_values, convergence_flag = compute_convergence_point(
#                 _tmp_libsizes, best_row, reg_func)
#
#             Kd_parts =best_row['Kd_CI'].split(',')
#             Kd_CI_lower= float(Kd_parts[0].strip('('))
#             Kd_CI_upper = float(Kd_parts[1].strip('()'))
#             Kd = float(best_row['Kd'])
#             Kd_best_status = '(Kd = {:.2f}, ci:({:.2f}, {:.2f}))'.format(Kd, Kd_CI_lower, Kd_CI_upper)
#             status.append(Kd_best_status)
#             norm_slopes = np.abs(slopes) / np.max(np.abs(slopes))
#
#             norm = Normalize(vmin=0, vmax=.05)
#             sm = ScalarMappable(cmap="Greys", norm=norm)
#             for i in range(len(_tmp_libsizes) - 1):
#                 ax.plot(_tmp_libsizes[i:i + 2], reg_func(_tmp_libsizes[i:i + 2], **best_row),
#                         color=sm.to_rgba(norm_slopes[i]), alpha=1, linewidth=2, zorder=1)
#
#         else:
#             # If no models pass the streak threshold, select the one with the longest streak
#             best_model_idx = eval_df['max_streak'].idxmax()
#             max_streak_data_d = max_streak_data_lst[best_model_idx]
#             best_model = eval_df.loc[best_model_idx]
#             best_row = best_model.to_dict()
#
#             Kd_parts = best_row['Kd_CI'].split(',')
#             Kd_CI_lower = float(Kd_parts[0].strip('('))
#             Kd_CI_upper = float(Kd_parts[1].strip('()'))
#             Kd = float(best_row['Kd'])
#             Kd_best_status = '(Kd = {:.2f}, ci:({:.2f}, {:.2f}))'.format(Kd, Kd_CI_lower, Kd_CI_upper)
#             print(Kd_best_status)
#             status.append(Kd_best_status)
#
#             status = ['Fail']+status
#             status.append(f'max streak: {eval_df.loc[best_model_idx]["max_streak"]}')
#             color = 'purple'
#             cancel = True
#             details_d = best_model.to_dict()
#
#             convergence_libsize_min = min(max_streak_data_d['libsize'])#[convergence_libsize_min_ind]
#             convergence_libsize_max = max(max_streak_data_d['libsize'])#[convergence_libsize_max_ind]
#             convergence_rho_min = np.mean(max_streak_data_d['rho'])
#             convergence_rho_max = np.mean(max_streak_data_d['rho'])#[convergence_libsize_max_ind]
#
#             details_d.update({
#                 'convergence_libsize_min':convergence_libsize_min,
#                 'convergence_libsize_max':convergence_libsize_max,
#                 'convergence_rho_min':convergence_rho_min,
#                 'convergence_rho_max':convergence_rho_max,
#                 'surr_libsize_min':max_streak_data_d['surr_libsize_min'],
#                 'surr_libsize_max':max_streak_data_d['surr_libsize_min']+surr_testing_window,
#                 'surr_rho_min':max_streak_data_d['surr_rho_min'],
#                 'surr_rho_max':max_streak_data_d['surr_rho_max']})
#
#         max_streak_data_d['libsize'] = np.array(max_streak_data_d['libsize'])
#         lib_inds = np.argwhere(max_streak_data_d['libsize'] <= convergence_libsize_max)
#         for ip, in lib_inds:  # libsize in enumerate(max_streak_data_d['libsize']):
#             libsize = max_streak_data_d['libsize'][ip]
#             pred_rho = max_streak_data_d['rho_pred'][
#                 ip]  # reg_func(libsize, **{key:best_row[key] for key in func_config_d.keys()})
#             plt.errorbar(libsize, pred_rho,
#                          yerr=pred_rho * percent_threshold,  # np.ravel(convergence_rhos),
#                          #              # yerr=[[row[metric] - metric_error], [ row[metric]+metric_error]],
#                          fmt='.', alpha=.5, ms=1, label='pred values',
#                          color=color,
#                          ecolor=color, elinewidth=1, capsize=1.5, zorder=500)
#
#             rho = max_streak_data_d['rho'][ip]  # libsize_sub['rho'].quantile(.5)#.mean()
#             # real_df_rel_rho_l, real_df_rel_rho_u = get_u_l(real_df_rel_rho_m, percent_threshold=percent_threshold)
#             plt.errorbar(libsize, rho,
#                          yerr=rho * percent_threshold, label='real values',  # np.ravel(convergence_rhos),
#                          fmt='.', alpha=.5, ms=1,
#                          color='k',
#                          ecolor='k', elinewidth=1, capsize=1.5, zorder=500)
#
#
#         if cancel is True:
#             if 'Pass' not in status:
#                 if 'Fail' not in status:
#                     status = ['Fail'] +status
#         print(status)
#         ax.text(.1, .15, '\n'.join(status),  # f'no streak pairs, check={len(convergence_libsize_saved)}',
#                             horizontalalignment='left',
#                             verticalalignment='top',
#                             transform=ax.transAxes)
#
#         details_d['status'] = status
#         copy_grp_d = grp_d.copy()
#         copy_grp_d['relation'] = rel
#         details_d.update(copy_grp_d)
#         convergence_grps.append(details_d)
#     xlims = ax.get_xlim()
#     ax.set_xlim([knn, xlims[1]])
#     fig_name = f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}__libsize_rho_{metric}__{str(grp_d["group_id"])}.png'
#
#     if 'Pass' in status:
#         spec_fig_dir = spec_fig_dir /'pass'
#     else:
#         spec_fig_dir = spec_fig_dir / 'fail'
#     spec_fig_dir.mkdir(exist_ok=True, parents=True)
#
#     fig.savefig(spec_fig_dir / fig_name, dpi=300, bbox_inches='tight')
#     print('fig saved', spec_fig_dir / fig_name, file=sys.stdout, flush=True)
#     plt.close()
#
#     snippet_csv = convergence_dir_csv_parts / 'convergence_grp{}_ind{}.csv'.format(int(grp_d['group_id']), ind)
#     existing_csv = [file for file in os.listdir(convergence_dir_csv_parts) if
#                     file.startswith('convergence_grp{}_ind'.format(int(grp_d['group_id'])))]
#     if len(convergence_grps) > 0:
#         print('convergence_grps', snippet_csv, file=sys.stdout, flush=True)
#         new_data_df = pd.DataFrame(convergence_grps)
#         if override_flag in [True, 'True']:
#             new_data_df.to_csv(snippet_csv)
#         else:
#             snippets = []
#             if len(existing_csv) > 0:
#                 for csv in existing_csv:
#                     snippets.append(pd.read_csv(convergence_dir_csv_parts / csv, index_col=0))
#             snippets.append(new_data_df)
#             pd.concat(snippets).to_csv(snippet_csv, index=False)
#         return new_data_df
#     else:
#         return
#
#     # return pd.DataFrame(convergence_grps)#

#
# def calc_binding_reg(df_sub, function_flag):
#
#     def inverse_exponential(x, L, x_initial, c, rho_initial):
#         return inverse_exponential_func(x, L, x_initial, c, rho_initial)
#
#     def binding(x, bmax, x_initial, kd, rho_initial):
#         return binding_func(x, bmax, x_initial, kd, rho_initial)
#
#     def f(x, bmax, x0, kd, rho_0):
#         # Ensure that L is always greater than or equal to L_0
#         L_adjusted = np.maximum(x - x0, 0)
#         return rho_0 + (bmax - rho_0) * (L_adjusted) / (kd + L_adjusted)
#
#     # Your data (example)
#     # sample for middle IQR:
#
#     knn = 20  # df_sub['knn'].max()
#     pctile_range = [.25, .75]
#     sample_kwargs = {'n': 40, 'replace': False}
#     iqr_dfs = []
#     filter_var = 'rho'
#     groupby_var = 'LibSize'
#     surr_testing_window = 75
#     # df_sub= df_sub[df_sub['LibSize']>=].copy()
#     for rel, real_df_full_rel in df_sub.groupby('relation'):
#         if len(real_df_full_rel) == 0:
#             continue
#
#         samples_df_list, samples_flags = bootstrap_raw_output(real_df_full_rel, filter_var, groupby_var,
#                                                               pctile_range, sample_kwargs=sample_kwargs)
#         iqr_dfs.extend(samples_df_list)
#
#     df_sub = check_empty_concat(iqr_dfs)
#     if len(df_sub) == 0:
#         print('no real data found', grp_d, file=sys.stderr, flush=True)
#         return
#
#     lib_sizes = df_sub['LibSize'].values
#     rhos = df_sub['rho'].values#np.array([100, 200, 300, 400, 500, 600])  # Replace with your actual data
#     # if df_sub.groupby('LibSize')['rho'].mean().min()<df_sub.groupby('LibSize')['rho'].quantile(.5).min():
#     #     rho_0_df = df_sub[['LibSize', 'rho']].groupby('LibSize').agg(lambda x: x.mean())
#     #     minrho_ind = np.argmin(rho_0_df['rho'].values)
#     #     rho_0 = max(rho_0_df['rho'].values[minrho_ind], 0)
#     #     libsize_min = rho_0_df.index[minrho_ind]
#     #     # print('mean', 'minrho_ind', minrho_ind)
#     #     # libsize_min = df_sub.groupby('LibSize')['LibSize'].mean().values[minrho_ind]
#     #     # print('mean', 'libsize_min', libsize_min)
#     # else:
#     #     rho_0_df = df_sub[['LibSize', 'rho']].groupby('LibSize').agg(lambda x: x.quantile(.5))
#     #     minrho_ind = np.argmin(rho_0_df['rho'].values)
#     #     rho_0 = max(rho_0_df['rho'].values[minrho_ind], 0)
#     #     libsize_min = rho_0_df.index[minrho_ind]
#
#     libsize_min = knn
#     # print('min', df_sub.head())
#     rho_0 = max(df_sub[df_sub['LibSize']<=libsize_min+1]['rho'].quantile(.5), 0)
#
#     if len(lib_sizes)<1:
#         print('no data', df_sub.head(), file=sys.stderr, flush=True)
#         return
#
#     if function_flag == 'inverse_exponential':
#         # Fit the logistic curve to the data
#         initial_guess = [.2, min(lib_sizes), 0.04, rho_0]
#         params, covariance = curve_fit(inverse_exponential, lib_sizes, rhos, p0=initial_guess,
#                                        bounds=[(-np.inf, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf)])
#
#         # Extract the fitted parameters
#         L, x0, k, off = params
#
#
#         # print(f"Fitted parameters:\nL = {L}\nk = {k}\nx0 = {x0}")
#
#         # Calculate the standard errors of the parameters (square roots of the diagonal of the covariance matrix)
#         param_errors = np.sqrt(np.diag(covariance))
#         # print(f"Standard errors:\nL = {param_errors[0]}\nk = {param_errors[1]}\nx0 = {param_errors[2]}")
#
#         # Calculate confidence intervals
#         alpha = 0.1  #05  # 95% confidence interval
#         n = len(lib_sizes)  # number of data points
#         p = len(params)  # number of parameters
#         dof = max(0, n - p)  # degrees of freedom
#
#         # t-value for the confidence interval
#         t_val = t.ppf(1 - alpha / 2, dof)
#
#         # Confidence intervals
#         ci = param_errors * t_val
#
#         fit_results = {
#             "max_lib_size": max(lib_sizes),
#             "L": L,
#             "k": k,
#             "x0": x0,
#             "off": off,
#             "L_error": param_errors[0],
#             "k_error": param_errors[1],
#             "x0_error": param_errors[2],
#             "L_CI": (L - ci[0], L + ci[0]),
#             "k_CI": (k - ci[1], k + ci[1]),
#             "x0_CI": (x0 - ci[2], x0 + ci[2])
#         }
#
#     elif function_flag == 'binding':
#         initial_guess = [.2, libsize_min, 7, rho_0]
#         # print('initial_guess', initial_guess)
#         bounds = ([0, libsize_min-2 , 0, min(-.01 if rho_0==0 else rho_0 * 0.5, rho_0 * 0.5)],
#                   [1, libsize_min + 2, 400, max(rho_0 * 1.5,.05)])
#         # print(bounds)
#
#         params, covariance = curve_fit(binding, lib_sizes, rhos,p0=initial_guess, bounds=bounds)
#         # print(f"Fitted parameters:\nL = {L}\nk = {k}\nx0 = {x0}")
#         L, x0, Kd, off = params
#         if np.isnan(L) is True:
#             params, covariance = curve_fit(binding, lib_sizes, rhos)
#             L, x0, Kd, off = params
#             print('nan', params, file=sys.stderr, flush=True)
#
#         # Calculate the standard errors of the parameters (square roots of the diagonal of the covariance matrix)
#         param_errors = np.sqrt(np.diag(covariance))
#         # print(f"Standard errors:\nL = {param_errors[0]}\nk = {param_errors[1]}\nx0 = {param_errors[2]}")
#
#         # Calculate confidence intervals
#         alpha = 0.05  # 05  # 95% confidence interval
#         n = len(lib_sizes)  # number of data points
#         p = len(params)  # number of parameters
#         dof = max(0, n - p)  # degrees of freedom
#
#         # t-value for the confidence interval
#         t_val = t.ppf(1 - alpha / 2, dof)
#
#         # Confidence intervals
#         ci = param_errors * t_val
#         # except:
#         #     print('error with curve fit errors', file=sys.stderr, flush=True)
#         #     L, Kd, x0, off = [np.nan, np.nan, np.nan, np.nan]
#         #     param_errors = [np.nan, np.nan, np.nan, np.nan]
#         #     ci = [np.nan, np.nan, np.nan, np.nan]
#
#         fit_results = {
#             "max_lib_size": max(lib_sizes),
#             "min_lib_size": libsize_min,
#             "L": L,
#             "Kd": Kd,
#             "x0": x0,#min(lib_sizes),
#             "off": off,# rho_0,
#             "L_error": param_errors[0],
#             "Kd_error": param_errors[1],
#             "x0_error": param_errors[2],
#             "L_CI": (L - ci[0], L + ci[0]),
#             "Kd_CI": (Kd - ci[1], Kd + ci[1]),
#             "x0_CI": (x0 - ci[2], x0 + ci[2]),
#         }
#         # print('fit_results', fit_results)
#
#     return fit_results

from numpy.polynomial import polynomial as P
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def process_group(arg_tuple):

    def f(x, bmax, x0, kd, rho_0):
        # Ensure that L is always greater than or equal to L_0
        L_adjusted = np.maximum(x - x0, 0)
        return rho_0 + (bmax - rho_0) * (L_adjusted) / (kd + L_adjusted)

    def filter_ptile(group, l=.25, u=.75):
        Q1 = group['rho'].quantile(l)
        Q3 = group['rho'].quantile(u)
        IQR = Q3 - Q1
        return group[(group['rho'] >= Q1) & (group['rho'] <= Q3)]

    def fit_binding(f, x_data, y_data, initial_guess, bounds):
        popt, pcov = curve_fit(f, x_data, y_data, p0=initial_guess, bounds=bounds)
        bmax_opt, x0_opt, kd_opt, rho_0_opt = popt

        param_errors = np.sqrt(np.diag(pcov))
        # print(f"Standard errors:\nL = {param_errors[0]}\nk = {param_errors[1]}\nx0 = {param_errors[2]}")

        # Calculate confidence intervals
        alpha = 0.05  # 05  # 95% confidence interval
        n = len(y_data)  # number of data points
        p = len(popt)  # number of parameters
        dof = max(0, n - p)  # degrees of freedom

        # t-value for the confidence interval
        t_val = t.ppf(1 - alpha / 2, dof)

        # Confidence intervals
        ci = param_errors * t_val

        fit_results = {
            "max_lib_size": max(x_data),
            "min_lib_size": min(x_data),
            "L": bmax_opt,
            "Kd": kd_opt,
            "x0": x0_opt,  # min(lib_sizes),
            "off": rho_0_opt,  # rho_0,
            "L_error": param_errors[0],
            "Kd_error": param_errors[1],
            "x0_error": param_errors[2],
            "L_CI": (bmax_opt - ci[0], bmax_opt + ci[0]),
            "Kd_CI": (kd_opt - ci[1], kd_opt + ci[1]),
            "x0_CI": (x0_opt - ci[2], x0_opt + ci[2]),

        }
        return popt, pcov, fit_results

    filter_center = lambda x: filter_ptile(x, .1, .9)
    filter_high = lambda x: filter_ptile(x, .9, .975)

    (grp_d, ind, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
     function_flag, plot_flag, override_flag, write_flag, datetime_flag, percent_threshold) = arg_tuple

    print('override_flag', override_flag)

    start_time = datetime.datetime.now()
    print(f'grp_id:{str(grp_d["group_id"])} received', start_time, file=sys.stdout, flush=True)
    calc_metric_csvs = calc_convergence_dir /config.calc_convergence_dir.dirs.csvs  # 'csvs2'
    metric_df_path = calc_metric_csvs / f'{str(grp_d["group_id"])}.csv'
    fig_dir = convergence_dir_csv_parts.parent /config.calc_convergence_dir.dirs.figures  # calc_metric_dir / 'figures2'
    spec_fig_dir = fig_dir / f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}' / f'Tp{grp_d["Tp"]}'  #/f'lag{grp_d["lag"]}'
    spec_fig_dir.mkdir(exist_ok=True, parents=True)

    weighted_flag = get_weighted_flag(grp_d)
    pal = config.pal.to_dict()
    surr_testing_window=75
    _max_libsize = 325

    metric = 'L_convergence_detail'
    fig_name = f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}_E{str(grp_d["E"])}_tau{grp_d["tau"]}_{weighted_flag}__libsize_rho_{metric}__{str(grp_d["group_id"])}.png'

    if override_flag in [False, 'False']:
        if write_flag in [False, 'False', None, 'append']:
            if (spec_fig_dir / fig_name).exists():
                print(f'figure: grp_id:{str(grp_d["group_id"])} already exists, skipping', file=sys.stdout, flush=True)
                return

    # Check if the metric dataframe already exists and if it should be overridden
    choice = check_for_exit(True, datetime_flag, metric_df_path, grp_d)

    meta_variables = ['tau', 'E', 'train_len', 'train_ind_i', 'knn', 'Tp_flag',
                      'Tp', 'lag', 'Tp_lag_total', 'sample', 'weighted', 'target_var',
                      'col_var', 'surr_var', 'col_var_id', 'target_var_id']
    # collect data
    if real_dfs_references is None:
        grp_path = calc_location / 'calc_refactor'/f'{grp_d["col_var_id"]}_{grp_d["target_var_id"]}'/ f'E{grp_d["E"]}_tau{grp_d["tau"]}'
        print('grp_path', grp_path, file=sys.stderr, flush=True)
        files = [file for file in os.listdir(grp_path) if (file.endswith('.csv')) & ('neither' in file)]
        print('files', files, file=sys.stderr, flush=True)
        real_df_full = pd.read_csv(grp_path / files[0])
        print(real_df_full.head(), file=sys.stderr, flush=True)
    else:
        real_df_full = collect_raw_data(real_dfs_references, meta_vars=meta_variables)
    real_df_full = real_df_full[real_df_full['LibSize'] >= grp_d['knn']].copy()
    real_df = real_df_full[real_df_full['LibSize'] <= _max_libsize].copy()
    if len(real_df_full) == 0:
        print('no real data found', grp_d, file=sys.stderr, flush=True)
        return

    target_relation = None
    if config.has_nested_attribute('calc_criteria_rates2.target_relation'):
        target_relation = config.calc_convergence_reg.target_relation
        real_df = real_df[real_df['relation'] == target_relation].copy()
    real_df = real_df[~real_df['relation'].str.isnumeric()].copy()

    if len(real_df) < 1:
        end_time = datetime.datetime.now()
        print(f'data for available: {len(real_df)}, grp_id: {grp_d["group_id"]}', grp_d, 'time elapsed: ',
              end_time - start_time, file=sys.stderr, flush=True)
        return

    fit_list = []
    for rel, rel_df in real_df.groupby('relation'):

        if len(rel_df) < 10:
            print('not enough data for real df', rel, file=sys.stderr, flush=True)
            continue
        else:
            # try:
            rel_df[['rho', 'LibSize']] = rel_df[['rho', 'LibSize']].astype(float)
            def_len = 700
            bounds = ([-np.inf, -np.inf, 1, -np.inf], [np.inf, np.inf, np.inf, np.inf])
            if 'erb' in [grp_d['col_var_id'], grp_d['target_var_id']]:
                initial_guess = [.4, 19, 25, .4]
            else:
                initial_guess = [.2, 19, 25, .2]
            libsize_max=325
            color = pal[rel]
            stop=False
            for ip in range(5):

                if stop is True:
                    continue
                else:
                    if ip>0:
                        fit_list = []

                    rel_df = rel_df.groupby('LibSize').apply(lambda x: x.sample(n=800, replace=True)).reset_index(drop=True)

                    flag_d = {'end_behavior': [], 'core_behavior': [], 'top_behavior': [], 'interval': []}

                    ## fit upper side of envelope
                    filtered_df = rel_df[(rel_df['LibSize'] <= 250)].copy()
                    sample_df = filtered_df.groupby('LibSize').apply(filter_high).reset_index(drop=True)
                    x_data = sample_df['LibSize']
                    y_data = sample_df['rho']

                    # Fit the function to the data
                    try:
                        popt_upper, pcov_upper, fit_results_upper = fit_binding(f, x_data, y_data, initial_guess, bounds)
                    except:
                        print(f'error with curve fit, E={grp_d["E"]}, tau={grp_d["tau"]}', file=sys.stderr, flush=True)
                        continue

                    x_fitted = np.arange(0, 350)  # EW_test.var1_xmap_var2['LibSize'].unique()
                    y_fitted_upper = f(x_fitted, *popt_upper)

                    # Extract the optimized parameters
                    bmax_opt_upper, x0_opt_upper, kd_opt_upper, rho_0_opt_upper = popt_upper
                    print(
                        f"Optimized parameters from above: bmax: {bmax_opt_upper}, x0: {x0_opt_upper}, kd: {kd_opt_upper}, rho_0: {rho_0_opt_upper}")
                    if rho_0_opt_upper * .975 >= bmax_opt_upper * 1.025:
                        flag_d['top_behavior'].append('converge from above')
                    elif f(x_fitted[0], *popt_upper) * .975 >= bmax_opt_upper * 1.025:
                        flag_d['top_behavior'].append('converge from above')

                    fit_results_upper.update(grp_d)
                    fit_results_upper['curve_fit'] = 'upper_envelope'
                    fit_results_upper['relation'] = rel
                    for flag in flag_d:
                        fit_results_upper[flag]='\n'.join(flag_d[flag]).strip('\n') if len(flag_d[flag])>0 else None
                    fit_list.append(fit_results_upper)

                    ## Fit the middle of the envelope
                    # Group by 'LibSize' and apply the filter
                    filtered_df = rel_df[(rel_df['LibSize'] <= libsize_max)].copy()
                    sample_df = filtered_df.groupby('LibSize').apply(filter_center).reset_index(drop=True)
                    x_data = sample_df['LibSize']  # np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # Independent variable
                    y_data = sample_df['rho']  # np.array([0.2, 0.6, 1.1, 2.1, 3.8, 4.9])   # Dependent variable

                    # Fit the function to the data
                    try:
                        popt_middle, pcov_middle, fit_results_middle = fit_binding(f, x_data, y_data, initial_guess, bounds)
                    except:
                        print(f'error with middle curve fit, E={grp_d["E"]}, tau={grp_d["tau"]}', file=sys.stderr, flush=True)
                        continue

                    fit_results_middle.update(grp_d)
                    fit_results_middle['curve_fit'] = 'middle_envelope'
                    fit_results_middle['relation'] = rel

                    # Extract the optimized parameters
                    bmax_opt, x0_opt, kd_opt, rho_0_opt = popt_middle

                    ## Evaluate
                    libsizes, rel_conv, rel_curve, rel_curve_interval = [], [], [], []
                    init_rho = f(20, *popt_middle)
                    for libsize in sample_df['LibSize'].unique():
                        rhos = sample_df[sample_df['LibSize'] == libsize].rho.values
                        libsizes.append(libsize)
                        rel_conv.append(len(rhos[rhos > bmax_opt]) / len(rhos))
                        rel_curve.append(len(rhos[rhos > f(libsize, *popt_middle)]) / len(rhos))
                        est_rho = f(libsize, *popt_middle)
                        delta_rho = np.abs(est_rho - init_rho) * percent_threshold
                        filtered_rhos = rhos[rhos > est_rho-delta_rho]
                        filtered_rhos = filtered_rhos[filtered_rhos < est_rho+delta_rho]
                        rel_curve_interval.append(len(filtered_rhos) / len(rhos))

                    metrics_df = pd.DataFrame({'LibSize': libsizes, 'rel_conv': rel_conv, 'rel_curve': rel_curve, 'rel_curve_interval':rel_curve_interval})
                    middle_df = metrics_df[((metrics_df['rel_curve'] >= .375) & (metrics_df['rel_curve'] <= .625)) | (metrics_df['rel_curve_interval']>=.50)].copy()
                    middle_distance_runner = middle_df.LibSize.max()
                    # print(middle_df)

                    convergence_ts = 7 * kd_opt + 20
                    y_fitted = f(x_fitted, *popt_middle)

                    ## Plotting
                    bins_x = len(filtered_df['LibSize'].unique())  # Number of bins for libsize (adjust as needed)
                    bins_y = 200  # Number of bins for rho (adjust as needed)
                    rho = filtered_df['rho'].values
                    libsize = filtered_df['LibSize'].values
                    title = f'E={grp_d["E"]}, tau={grp_d["tau"]}'

                    # h, xedges, yedges = np.histogram2d(libsize, rho, bins=[bins_x, bins_y])
                    #
                    # # Plot 2D histogram with log color scale
                    # fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
                    # pcm1 = ax.pcolormesh(xedges, yedges, h.T, cmap='plasma', norm=LogNorm()) # Log scale
                    # fig.colorbar(pcm1, ax=ax, label='# points')
                    # ax.set_xlabel('LibSize')
                    # ax.set_ylabel(r'$\rho$')
                    #
                    # # Plot the annotations
                    # ax.axvline(convergence_ts, color=color, linestyle='--', label=f'convergence ts={convergence_ts:.2f}')
                    # ax.axvline(middle_df.LibSize.max(), color='teal', linestyle='--', label=f'upper limit of centered binding curve {middle_df.LibSize.max()}')
                    #
                    # ax.plot(x_fitted, y_fitted, color=color,
                    #         label=f"central binding curve; ct={kd_opt * 7 + 20:.2f}, rho_f={bmax_opt:.2f}")
                    # ax.plot(x_fitted, y_fitted_upper, color='k', linestyle='--', alpha=.5,
                    #         label=f">= 90%ile binding curve; ct={kd_opt_upper * 7 + 20:.2f}, rho_f={bmax_opt_upper:.2f}")
                    # ax.set_xlim([0, 350])
                    #
                    # Evaluate the convergence
                    counter = 0
                    while counter <2:
                        caveats = []
                        if counter ==0:
                            large_lib_rho = filtered_df[filtered_df['LibSize'] >= libsize_max - 20].rho
                            greater_than_zero = large_lib_rho.values[large_lib_rho.values > 0]
                        if counter ==1:
                            large_lib_rho = filtered_df[(filtered_df['LibSize'] >= middle_distance_runner-20) &
                                                        (filtered_df['LibSize'] <= middle_distance_runner)].rho
                            greater_than_zero = large_lib_rho.values[large_lib_rho.values > 0]

                        # no improvement
                        small_lib_rhos = filtered_df[filtered_df['LibSize'] < 40].rho.values[
                                filtered_df[filtered_df['LibSize'] < 40].rho.values > 0]
                        res = stats.ttest_ind(
                            greater_than_zero,
                            small_lib_rhos,
                            equal_var=False, alternative='greater')
                        if res.pvalue > .05:
                            caveats.append(
                                f'end behavior same mean, {np.mean(greater_than_zero):.3f}, {np.mean(small_lib_rhos):.3f}')
                            print(
                                f'p value, same mean, {np.mean(greater_than_zero):.3f}, {np.mean(small_lib_rhos):.3f}', res.pvalue)

                        # final rho mean not statistically different from zero
                        res = stats.ttest_ind_from_stats(large_lib_rho.mean(),
                                                         large_lib_rho.std(),
                                                         large_lib_rho.count(),
                                                         0,
                                                         large_lib_rho.std(),
                                                         large_lib_rho.count(), equal_var=True, alternative='greater')

                        if res.pvalue > .05:
                            caveats.append(f'end behavior, zero mean, {large_lib_rho.mean():.3f}')
                            print(f'p value, zero mean, {large_lib_rho.mean():.3f}', res.pvalue)
                        else:
                            res = stats.ttest_1samp(large_lib_rho.values, popmean=0, alternative='greater')
                            if res.pvalue > .05:
                                caveats.append('end behavior, zero popmean')
                                print('p value, zero mean', res.pvalue)
                            else:
                                # final rho mean not statistically different from zero
                                perc_geq_zero = len(greater_than_zero) / len(large_lib_rho)
                                if perc_geq_zero < .90:
                                    caveats.append(f'end behavior, perc greater than zero, {perc_geq_zero:.3f}')
                                    print('perc greater than zero', perc_geq_zero)

                        if len(caveats) > 0:
                            print('caveats', caveats)
                            if counter == 0:
                                flag_d['end_behavior'] = caveats
                            if counter == 1:
                                flag_d['core_behavior'] = caveats
                                counter=2
                            counter += 1
                        else:
                            counter = 2

                    end_behavior = ',\n'.join(flag_d['end_behavior']).strip(',\n') if len(flag_d['end_behavior']) > 0 else ''
                    core_behavior = ',\n'.join(flag_d['core_behavior']).strip(',\n') if len(flag_d['core_behavior']) > 0 else ''
                    top_behavior = ',\n'.join(flag_d['top_behavior']).strip(',\n') if len(flag_d['top_behavior']) > 0 else ''

                    cut_value = 25

                    tmp = middle_df.copy()#metrics_df[(metrics_df['rel_curve'] > .375) & (metrics_df['rel_curve'] < .625) & (
                                # metrics_df['LibSize'] <= middle_df.LibSize.max())].copy()
                    tmp = tmp.reset_index(drop=True)
                    tmp['delta'] = tmp['LibSize'].diff()

                    #test less strict
                    convergence_interval=None
                    libsizes = tmp['LibSize'].sort_values(ascending=False).values
                    for libsize in libsizes:
                        if convergence_interval is None:
                            less_than_df = tmp[tmp['LibSize']<=libsize].copy()
                            lower_edge_LibSize = libsize - surr_testing_window#less_than_df[
                                # less_than_df['LibSize'] <= (libsize - surr_testing_window)].LibSize.max()
                            less_than_df = less_than_df[(less_than_df['LibSize'] >= lower_edge_LibSize)].copy()
                            if less_than_df["delta"].mean() <= cut_value:
                                convergence_interval = (
                                    less_than_df.LibSize.max() - surr_testing_window, less_than_df.LibSize.max())
                                title += f'\nConvergence time: {convergence_ts:.2f}, average delta {less_than_df["delta"].mean():.3f}'
                            else:
                                print('E', grp_d['E'], 'tau', grp_d['tau'], 'less than', libsize, lower_edge_LibSize, less_than_df["delta"].mean())

                    if convergence_interval is None:
                        title += f'\nConvergence time: {convergence_ts:.3f} (interval too short), average delta {less_than_df["delta"].mean():.3f}'
                        flag_d['interval'].append('no convergence interval')


                    fit_results_middle['convergence_interval'] = convergence_interval

                    for flag in flag_d:
                        fit_results_middle[flag]='\n'.join(flag_d[flag]).strip('\n') if len(flag_d[flag])>0 else None
                    fit_results_middle.update(grp_d)
                    fit_list.append(fit_results_middle)

                    flag_count = sum([len(flag_d[flag]) for flag in ['end_behavior', 'core_behavior', 'top_behavior']])
                    if (convergence_interval is None) & (flag_count == 0):
                        stop = False
                    else:
                        stop = True
                        if len(fit_list)==2:
                            title = '[assessed] ' + title

                rho = filtered_df['rho'].values
                libsize = filtered_df['LibSize'].values
                h, xedges, yedges = np.histogram2d(libsize, rho, bins=[bins_x, bins_y])

                # Plot 2D histogram with log color scale
                fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
                pcm1 = ax.pcolormesh(xedges, yedges, h.T, cmap='plasma', norm=LogNorm())  # Log scale
                fig.colorbar(pcm1, ax=ax, label='# points')
                ax.set_xlabel('LibSize')
                ax.set_ylabel(r'$\rho$')

                # Plot the annotations
                ax.axvline(convergence_ts, color=color, linestyle='--',
                           label=f'convergence ts={convergence_ts:.2f}')
                ax.axvline(middle_df.LibSize.max(), color='teal', linestyle='--',
                           label=f'upper limit of centered binding curve {middle_df.LibSize.max()}')

                ax.plot(x_fitted, y_fitted, color=color,
                        label=f"central binding curve; ct={kd_opt * 7 + 20:.2f}, rho_f={bmax_opt:.2f}")
                ax.plot(x_fitted, y_fitted_upper, color='k', linestyle='--', alpha=.5,
                        label=f">= 90%ile binding curve; ct={kd_opt_upper * 7 + 20:.2f}, rho_f={bmax_opt_upper:.2f}")
                ax.set_xlim([0, 350])

                h, l = ax.get_legend_handles_labels()
                ax.legend(h, l, loc='upper left', bbox_to_anchor=(1.01, 1))
                ax.set_title('\n'.join([element for element in [title, top_behavior, core_behavior, end_behavior] if
                                        len(element) > 0]).strip('\n'))
                if 'erb' in fig_name:
                    ax.set_ylim([-.15, .7])
                else:
                    ax.set_ylim([-.15, .4])

                if convergence_interval is not None:
                    if len(end_behavior) > 0:
                        alpha = .5
                    else:
                        alpha = .2
                    ax.axvspan(convergence_interval[0], convergence_interval[1], color='k', alpha=alpha,
                               label='convergence interval')

    if convergence_interval is None:
        flag_count += 1

    if flag_count > 0:
        spec_fig_dir = spec_fig_dir / 'fail'
    else:
        spec_fig_dir = spec_fig_dir / 'pass'

    spec_fig_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(spec_fig_dir / fig_name, dpi=300, bbox_inches='tight')
    print('fig saved', spec_fig_dir / fig_name, file=sys.stdout, flush=True)
    plt.close()

    snippet_csv = convergence_dir_csv_parts / 'convergence_grp{}_ind{}.csv'.format(int(grp_d['group_id']), ind)
    existing_csv = [file for file in os.listdir(convergence_dir_csv_parts) if
                    file.startswith('convergence_grp{}_ind'.format(int(grp_d['group_id'])))]
    if len(fit_list) > 0:
        print('convergence_grps', snippet_csv, file=sys.stdout, flush=True)
        new_data_df = pd.DataFrame(fit_list)
        if write_flag in ['replace']:#[True, 'True']:
            new_data_df.to_csv(snippet_csv)
        else:
            snippets = []
            if len(existing_csv) > 0:
                for csv in existing_csv:
                    snippets.append(pd.read_csv(convergence_dir_csv_parts / csv, index_col=0))
            snippets.append(new_data_df)
            pd.concat(snippets).to_csv(snippet_csv, index=False)
        return new_data_df
    else:
        return

# def parse_flags(args, default_percent_threshold=.05, default_function_flag='binding',
#                                        default_res_flag='', default_second_suffix=''):
#
#     percent_threshold = None
#     function_flag = None
#     res_flag = None
#     second_suffix = default_second_suffix
#
#     if isinstance(args.flags, list):
#         if 'inverse_exponential' in args.flags:
#             function_flag = 'inverse_exponential'
#         elif 'binding' in args.flags:
#             function_flag = 'binding'
#
#         for flag in args.flags:
#             if 'coarse' in flag:
#                 res_flag = '_' + flag
#
#         numeric_flags = [is_float(val) for val in args.flags if is_float(val) is not None]
#         if len(numeric_flags) > 0:
#             percent_threshold_candidates = [flag for flag in numeric_flags if flag <= 1]
#             if len(percent_threshold_candidates) > 0:
#                 percent_threshold = percent_threshold_candidates[0]
#
#             second_suffix_candidates = [flag for flag in numeric_flags if flag > 1]
#             if len(second_suffix_candidates) > 0:
#                 second_suffix = f'_{int(second_suffix_candidates[0])}'
#
#     if function_flag is None:
#         function_flag = default_function_flag
#     if percent_threshold is None:
#         percent_threshold = default_percent_threshold
#     if res_flag is None:
#         res_flag = default_res_flag
#
#     return percent_threshold, function_flag, res_flag, second_suffix

# def construct_convergence_name(args, carc_config_d, percent_threshold, second_suffix):
#
#     percent_threshold_label = str(percent_threshold * 100).lstrip('.0').replace('.', 'p')
#     if '.' in percent_threshold_label:
#         percent_threshold_label = '_' + percent_threshold_label.replace('.', 'p')
#
#     calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}/tolerance{percent_threshold_label}'
#     if isinstance(args.dir, str):
#         if len(args.dir) > 0:
#             subdir = args.dir
#             calc_convergence_dir_name = f'{calc_convergence_dir_name}/{subdir}{second_suffix}'
#     else:
#         calc_convergence_dir_name = f'{calc_convergence_dir_name}{second_suffix}'
#
#     return calc_convergence_dir_name


if __name__ == '__main__':

    delta_label = 'r50p-s50p'
    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # override= False
    override = args.override #if args.override not in [None else False
    write = args.write
    if override is False:
        if write in ['append', None]:
            write = 'append'
    #     override = args.override
    # else:
    #     override = False

    # second_suffix = ''
    # if args.test:
    second_suffix = f'_{int(time.time() * 1000)}' if args.test is True else ''

    # percent_threshold = .05
    # function_flag = 'binding'
    # res_flag = ''

    # def construct_convergence_dir_name(args, carc_config_d, percent_threshold=.05, function_flag='binding', res_flag=''):
    #
    #     if isinstance(args.flags, list):
    #         if 'inverse_exponential' in args.flags:
    #             function_flag = 'inverse_exponential'
    #         elif 'binding' in args.flags:
    #             function_flag = 'binding'
    #
    #         for flag in args.flags:
    #             if 'coarse' in flag:
    #                 res_flag= '_'+flag
    #
    #         numeric_flags= [is_float(val) for val in args.flags if is_float(val) is not None]
    #         if len(numeric_flags) > 0:
    #             percent_threshold_candidates = [flag for flag in numeric_flags if flag<=1]
    #             if len(percent_threshold_candidates) > 0:
    #                 percent_threshold = percent_threshold_candidates[0]
    #
    #             second_suffix_candidates = [flag for flag in numeric_flags if flag > 1]
    #             if len(second_suffix_candidates) > 0:
    #                 second_suffix = f'_{int(second_suffix_candidates[0])}'
    #
    #     percent_threshold_label = str(percent_threshold * 100).lstrip('.0').replace('.', 'p')
    #     if '.' in percent_threshold_label:
    #         percent_threshold_label = '_' + percent_threshold_label.replace('.', 'p')
    #
    #     calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}/tolerance{percent_threshold_label}'
    #     if isinstance(args.dir, str):
    #         if len(args.dir) > 0:
    #             subdir = args.dir
    #             calc_convergence_dir_name = f'{calc_convergence_dir_name}/{subdir}{second_suffix}'
    #     else:
    #         calc_convergence_dir_name = f'{calc_convergence_dir_name}{second_suffix}'
    #     return calc_convergence_dir_name


    # print(percent_threshold, function_flag, res_flag)
    # if isinstance(args.dir, str):
    #     if len(args.dir) > 0:
    #         parent_dir = Path('approach2')/args.dir
    # else:
    #     parent_dir = Path('approach2')

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')

    calc_carc_mirrored = proj_dir / config.mirrored.calc_carc
    carc_config_d = config.calc_carc_dir
    # calc_convergence_dir_name = carc_config_d.dirs.calc_convergence_dir  #config['calc_criteria_rates2']['calc_metrics_dir']['name']#'calc_metrics2'

    calc_log_path_new = calc_carc_mirrored / f'{carc_config_d.csvs.completed_runs_csv}.csv'
    calc_log2_df = pd.read_csv(calc_log_path_new, index_col=0, low_memory=False)

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    percent_threshold, function_flag, res_flag, second_suffix = parse_flags(args,
                                                                                 default_percent_threshold=.05,
                                                                                 default_function_flag='binding',
                                                                                 default_res_flag='',
                                                                                 default_second_suffix=second_suffix)

    calc_convergence_dir_name = construct_convergence_name(args, carc_config_d, percent_threshold, second_suffix)

    # calc_convergence_dir_name = f'{carc_config_d.dirs.calc_convergence_dir}/tolerance{percent_threshold_label}'
    calc_convergence_dir = calc_location / calc_convergence_dir_name
    # if isinstance(args.dir, str):
    #     if len(args.dir) > 0:
    #         subdir = args.dir
    #         calc_convergence_dir = calc_location / carc_config_d.dirs.calc_convergence_dir / f'{subdir}{second_suffix}'
    # else:
    #     calc_convergence_dir = calc_location / f'{carc_config_d.dirs.calc_convergence_dir}{second_suffix}'

    # calc_convergence_dir = calc_location / calc_convergence_dir_name #/ f'{function_flag}{res_flag}{second_suffix}'
    calc_convergence_dir.mkdir(exist_ok=True, parents=True)
    calc_convergence_dir_csvs = calc_convergence_dir /config.calc_convergence_dir.dirs.csvs

    # perc_threshold_dir = calc_convergence_dir / f'{percent_threshold_label}'/parent_dir
    # perc_threshold_dir = calc_convergence_dir

    convergence_dir_csv_parts = calc_convergence_dir / f'{config.calc_convergence_dir.dirs.summary_frags}'
    calc_convergence_dir_csvs.mkdir(exist_ok=True, parents=True)
    convergence_dir_csv_parts.mkdir(exist_ok=True, parents=True)
    fig_dir = calc_convergence_dir /f'{config.calc_convergence_dir.dirs.figures}'
    fig_dir.mkdir(exist_ok=True, parents=True)

    query_keys = config.calc_criteria_rates2.query_keys
    if 'knn' not in query_keys:
        query_keys.append('knn')


    if args.group_file is not None:
        grp_csv = f'{args.group_file}.csv'
    else:
        grp_csv = f'{carc_config_d.csvs.calc_grp_run_csv}.csv'

    calc_grps_path = calc_carc_mirrored / grp_csv
    calc_grps_df = pd.read_csv(calc_grps_path)
    calc_grps_df = calc_grps_df[calc_grps_df['weighted'] == False].copy()

    plot_flag = True
    datetime_flag = None  #datetime.datetime(2024, 8, 31,6, 20 )

    # Determine the number of CPUs to use from the SLURM environment variable
    if Path('/Users/jlanders').exists():
        print('running locally', file=sys.stdout, flush=True)

        convergence_grps = []
        delta_grps = []
        calc_grps_df = calc_grps_df[
                                    # (calc_grps_df['col_var_id'] == 'erb') &
                                    (calc_grps_df['target_var_id'] == 'wu') &
                                    (calc_grps_df['train_ind_i'] == 0) &
                                    (calc_grps_df['E'] == 4) &
                                    (calc_grps_df['Tp'] == 20)].copy()
        ind = 0
        for authors, auth_grp in calc_grps_df.groupby(['col_var_id', 'target_var_id']):
            print(f'authors: {authors}', file=sys.stdout, flush=True)
            auth_pair = f'{authors[0]}_{authors[1]}'
            calc_grps = auth_grp.to_dict(orient='records')

            calc_log2_df = calc_log2_df[~calc_log2_df['df_path'].str.contains('_pctiles')]
            arg_tuples = []

            for grp_d in calc_grps:
                query_d = {}
                for key in query_keys:
                    if isinstance(grp_d[key], str):
                        query_d[key] = f'"{grp_d[key]}"'
                    else:
                        query_d[key] = grp_d[key]

                query_string = [f'{key}=={value}' for key, value in query_d.items()]
                query_string = ' & '.join(query_string)
                grp_df = calc_log2_df.query(query_string)

                weighted_flag = get_weighted_flag(grp_d)
                if weighted_flag=='weighted':
                    grp_df = grp_df[grp_df['weighted'] == True].copy()
                elif weighted_flag=='unweighted':
                    grp_df = grp_df[grp_df['weighted'] == False].copy()

                # Filter for real data frames
                real_dfs_references = grp_df[grp_df['surr_var'] == 'neither'].copy()
                if len(real_dfs_references)>0:
                    # arg_tuples.append((grp_d, ind, config, calc_convergence_dir,convergence_dir_csv_parts, real_dfs_references, function_flag,
                    #                plot_flag, override, datetime_flag, percent_threshold))
                    arg_tuples.append(
                        (grp_d, ind, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
                         function_flag,
                         plot_flag, override, write, datetime_flag, percent_threshold))
                    ind+=1
                # arg_tuples_poly.append((grp_d, config, calc_convergence_dir_poly, real_dfs_references,
                #                         plot_flag, override, datetime_flag))

            num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', min([4, len(arg_tuples)])))
            print('num_cpus:', num_cpus, file=sys.stdout, flush=True)
            #
            # # # Use multiprocessing to parallelize the process
            # with Pool(num_cpus) as pool:
            #     results = pool.map(process_group, arg_tuples)

            with Pool(num_cpus) as pool:
                results= pool.map(process_group, arg_tuples)

            results = [result for result in results if result is not None]

            if len(results) > 0:
                convergence_grps.append(pd.concat(results))

    else:
        if args.inds is not None:
            index = int(args.inds[-1])
        else:
            print('calc_criteria_rates2; index is required', file=sys.stdout, flush=True)
            sys.exit(0)
        # index = int(sys.argv[1])
        grp_ds = calc_grps_df.to_dict(orient='records')
        if index >= len(grp_ds):
            print('index out of range', file=sys.stdout, flush=True)
            sys.exit(1)
        else:
            grp_d = grp_ds[index]

        # calc_log2_df = pd.read_csv(calc_carc_mirrored / 'calc_log4.csv', index_col=0, low_memory=False)
        calc_log2_df = calc_log2_df[~calc_log2_df['df_path'].str.contains('_pctiles')]
        arg_tuples = []

        query_d = {}
        for key in query_keys:
            if isinstance(grp_d[key], str):
                query_d[key] = f'"{grp_d[key]}"'
            else:
                query_d[key] = grp_d[key]

        query_string = [f'{key}=={value}' for key, value in query_d.items()]
        query_string = ' & '.join(query_string)
        grp_df = calc_log2_df.query(query_string)

        real_dfs_references = grp_df[(grp_df['surr_var'] == 'neither') & (grp_df['weighted'].isin([False]))].copy()
        if len(real_dfs_references) > 0:
            arg_tuples.append(
                (grp_d, index, config, calc_convergence_dir, convergence_dir_csv_parts, real_dfs_references,
                 function_flag,
                 plot_flag, override, write, datetime_flag, percent_threshold))

        process_group(arg_tuples[0])
        result = process_group(arg_tuples[0])

        if len(result) > 0:
            print('convergence calculated successfully!', file=sys.stdout, flush=True)
        else:
            print('convergence not calculated', file=sys.stdout, flush=True)

    #SBATCH -p debug
