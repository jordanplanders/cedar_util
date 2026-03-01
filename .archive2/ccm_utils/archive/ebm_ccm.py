import pandas as pd

import numpy as np
import joblib
from pathlib import Path
import os
notebook_dir = Path(os.getcwd())/'notebooks'
proj_dir = Path(os.getcwd())#notebook_dir.parent
# print(proj_dir)
output_dir = notebook_dir/'output_files'
# print(output_dir)

import sys
# sys.path.append(str(proj_dir/'msccm_python'))

import find_acceptable_lib as fcl
import process_output as po
import archive2.ccm_utils.archive.ccm_helper as ch

from collections import defaultdict

import datetime as dt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def main():

    train_len = int(sys.argv[1])

    # Load the data
    df = pd.read_csv(output_dir/'vieira_tsi_ebm.csv', index_col=0)
    isospecsurr_df = pd.read_csv(output_dir/'isospec_surr_ebm_T.csv', index_col=0)
    arsurr_df = pd.read_csv(output_dir/'ar1_surr_ebm_T.csv', index_col=0)

    E = 5  # 13#Es[0]
    tau = -1  # taus[0]#, Tp_max,
    Tp_max = 2  # 5
    sample = 80

    target_var = 'temperature'
    col_var = 'solar_incoming'

    lib_opts = fcl.find_acceptable_libstart_maxlen(len(df['temperature']),
                                                   E, tau, Tp=Tp_max, l_max=train_len)
    lib_inds = np.arange(1, len(lib_opts.acceptablelib_inds), 8)

    lags = np.arange(-20, 21, 4)
    # len(lib_opts.acceptablelib_inds), len(lib_inds), lib_inds, lags


    try:
        _ccm_d_tmp_s1_ar = joblib.load(notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempar1_tsireal_{train_len}.joblib')
    except:
        _ccm_d_tmp_s1_ar = defaultdict(list)

    try:
        _ccm_d_tmp_s1_isospec = joblib.load(notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempisospec_tsireal_{train_len}.joblib')
    except:
        _ccm_d_tmp_s1_isospec = defaultdict(list)

    try:
        _ccm_d_tmp_real = joblib.load(notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempreal_tsireal_{train_len}.joblib')
    except:
        _ccm_d_tmp_real = defaultdict(list)

    start_train = dt.datetime.now()
    print('train len: ', train_len, 'start time: ', start_train)

    start_ts = dt.datetime.now()
    try:
        print('starting ar1', start_ts)
        _ccm_d_tmp_s1_ar = ch.CCM_lib_opts(E, tau, Tp_max, target_var, col_var, arsurr_df, train_len=train_len,
                                           lags=lags, knn=E, lib_opts=lib_opts, lib_inds=lib_inds, skip_0=True,
                                           ccm_d=_ccm_d_tmp_s1_ar, seed=777, sample=sample, libSizes=None)
        try:
            joblib.dump(_ccm_d_tmp_s1_ar, notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempar1_tsireal_{train_len}.joblib')
        except:
            print('saving ar1 failed')
    except:
        print('ar1 failed')
    end_ts = dt.datetime.now()
    print('ar1 time: ', end_ts - start_ts, end_ts)

    start_ts = dt.datetime.now()
    try:
        print('starting isospec', start_ts)
        _ccm_d_tmp_s1_isospec = ch.CCM_lib_opts(E, tau, Tp_max, target_var, col_var, isospecsurr_df,
                                                train_len=train_len, lags=lags, knn=E, lib_opts=lib_opts,
                                                lib_inds=lib_inds, skip_0=True, ccm_d=_ccm_d_tmp_s1_isospec, seed=777,
                                                sample=sample, libSizes=None)
        try:
            joblib.dump(_ccm_d_tmp_s1_isospec,
                        notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempisospec_tsireal_{train_len}.joblib')
        except:
            print('saving isospec failed')
    except:
        print('isospec failed')
    end_ts = dt.datetime.now()
    print('isospec time: ', end_ts - start_ts, end_ts)

    start_ts = dt.datetime.now()
    try:
        print('starting real', start_ts)
        _ccm_d_tmp_real = ch.CCM_lib_opts(E, tau, Tp_max, target_var, col_var, df, train_len=train_len, lags=lags,
                                          lib_opts=lib_opts, knn=E, lib_inds=lib_inds, skip_0=True,
                                          ccm_d=_ccm_d_tmp_real, seed=777, sample=sample, libSizes=None)
        try:
            joblib.dump(_ccm_d_tmp_real, notebook_dir / 'output_files' / f'ccm_vieira_f_ebm_tempreal_tsireal_{train_len}.joblib')
        except:
            print('saving real failed')
    except:
        print('real failed')
    end_ts = dt.datetime.now()
    print('real time: ', end_ts - start_ts, end_ts)

    end_train = dt.datetime.now()
    print('train time: ', end_train - start_train)

if __name__ == '__main__':
    main()

# echo 150 200 300 400 500 | xargs -n 1 -P 2 python3 -W ignore::RuntimeWarning msccm_python/ebm_ccm.py
# echo 150 | xargs -n 1 -P 1 python3 -W ignore::RuntimeWarning ebm_ccm.py