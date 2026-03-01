import numpy as np
from collections import defaultdict
import pandas as pd
import find_acceptable_lib as fcl
import process_output as po
import pyEDM as pe


def CCM_lib_opts(E, tau, Tp_max, target_var, col_var, df_raw, train_len=None, lags=None, lib_opts=None, lib_inds=None,
                 skip_0=True, ccm_d=None, seed=777, sample=100, libSizes=None, knn=5, **kwargs):
    """

    :param E:
    :param tau:
    :param Tp_max:
    :param target_var:
    :param col_var:
    :param df_raw:
    :param train_len:
    :param lags:
    :param lib_opts:
    :param lib_inds:
    :param skip_0:
    :param ccm_d:
    :param seed:
    :param sample:
    :param libSizes:
    :param knn:
    :param kwargs:
    :return:
    """

    num_libSizes = kwargs['num_libSizes'] if 'num_libSizes' in kwargs else None
    num_libs = kwargs['num_libs'] if 'num_libs' in kwargs else 10

    npartial = np.abs(tau) * (E - 1)

    if ccm_d is None:
        ccm_d = defaultdict(list)
    ccm_stash = []

    if lib_opts is None:
        if train_len is None:
            train_len = int(.6 * len(df_raw))  # first_insignificant_lag
        elif train_len < npartial * 2:
            train_len = npartial * 2
        lib_opts = fcl.find_acceptable_libstart_maxlen(len(df_raw),
                                                       E, tau, Tp=Tp_max, l_max=train_len)

    if lib_inds is not None:
        if max(lib_inds) > len(lib_opts.acceptablelib_inds) - 1:
            lib_inds = lib_inds[lib_inds <= len(lib_opts.acceptablelib_inds) - 1]
    else:
        if len(lib_opts.acceptablelib_inds) < num_libs:
            num_libs = len(lib_opts.acceptablelib_inds) - 1
        lib_inds = np.linspace(1, len(lib_opts.acceptablelib_inds) - 1, num_libs, dtype=int)

    if skip_0 is True:
        lib_inds = lib_inds[1:]

    if lags is None:
        lag_amplitude = max([Tp_max, 3])
        lag_spacing = max([1, .1 * lag_amplitude])
        lags = np.concatenate(
            [np.arange(0, 2 * max([Tp_max, 3]) + 1, lag_spacing), np.arange(-2 * max([Tp_max, 3]), 0, lag_spacing)],
            axis=0)
        np.sort(lags)

    if train_len is None:
        train_len = (len(df_raw) - lib_opts.acceptablelib_inds[-1]) - npartial

    for iq, lib_sub_ind in enumerate(lib_inds):

        train_ind_i = lib_opts.acceptablelib_inds[lib_sub_ind]
        train_ind_f = min([train_ind_i + train_len + npartial, len(df_raw)])
        df_sub = df_raw.iloc[train_ind_i:train_ind_f].copy().reset_index().drop(columns=['index'])
        if iq% 10 == 0:
            # print('lib_sub_ind', lib_sub_ind)
            print(iq, 'train_ind_i:train_ind_f', train_ind_i, train_ind_f)

        for lag in lags:
            shifted = df_sub.copy()
            shifted[target_var] = shifted[target_var].shift(lag)
            shifted = shifted.dropna()

            max_liblen = len(shifted) - npartial - Tp_max

            if libSizes is not None:
                if not isinstance(libSizes, np.ndarray):
                    libSizes = np.array(libSizes)
                libSizes = libSizes[libSizes >= E + 1]
                libSizes = libSizes[libSizes <= min([max_liblen, train_len])]
                libSizes = list(libSizes)
            else:
                if num_libSizes is None:
                    if min([max_liblen, train_len]) - max([2, E + 1]) < 40:
                        libSizes = list(np.arange(max([2, E + 1]), min([max_liblen, train_len]), 1))
                    else:
                        libSizes = list(np.linspace(max([2, E + 1]), min([max_liblen, train_len]), 40, dtype=int))
                else:
                    if min([max_liblen, train_len]) - max([2, E + 1]) > num_libSizes:
                        libSizes = list(
                            np.linspace(max([2, E + 1]), min([max_liblen, train_len]), num_libSizes, dtype=int))
                    else:
                        libSizes = list(np.arange(max([2, E + 1]), min([max_liblen, train_len]), 1))

            ccm_out = pe.CCM(dataFrame=shifted,
                             E=E, Tp=Tp_max, tau=tau, exclusionRadius=0, knn=knn, verbose=False,
                             columns=col_var, target=target_var, libSizes=libSizes,
                             sample=sample, embedded=False, seed=seed, includeData=True, returnObject=True,
                             showPlot=False, noTime=False)

            ccm_out_df = pd.concat(
                [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
            ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, train_ind_i, train_ind_f, lag=lag)

            ccm_stash.append(ccm_out_df)
            if isinstance(ccm_d, list):
                ccm_d.append(ccm_out_df)
            else:
                ccm_d[(E, tau)].append(ccm_out_df)
    return ccm_d#ccm_stash


def CCM_lib_opts(E, tau, Tp_max, target_var, col_var, df_raw, train_len=None, lags=None, lib_opts=None, lib_inds=None,
                 skip_0=True, ccm_d=None, seed=777, sample=100, libSizes=None, knn=5, **kwargs):
    """

    :param E:
    :param tau:
    :param Tp_max:
    :param target_var:
    :param col_var:
    :param df_raw:
    :param train_len:
    :param lags:
    :param lib_opts:
    :param lib_inds:
    :param skip_0:
    :param ccm_d:
    :param seed:
    :param sample:
    :param libSizes:
    :param knn:
    :param kwargs:
    :return:
    """

    num_libSizes = kwargs['num_libSizes'] if 'num_libSizes' in kwargs else None
    num_libs = kwargs['num_libs'] if 'num_libs' in kwargs else 10

    npartial = np.abs(tau) * (E - 1)

    if ccm_d is None:
        ccm_d = defaultdict(list)
    ccm_stash = []

    if lib_opts is None:
        if train_len is None:
            train_len = int(.6 * len(df_raw))  # first_insignificant_lag
        elif train_len < npartial * 2:
            train_len = npartial * 2
        lib_opts = fcl.find_acceptable_libstart_maxlen(len(df_raw),
                                                       E, tau, Tp=Tp_max, l_max=train_len)

    if lib_inds is not None:
        if max(lib_inds) > len(lib_opts.acceptablelib_inds) - 1:
            lib_inds = lib_inds[lib_inds <= len(lib_opts.acceptablelib_inds) - 1]
    else:
        if len(lib_opts.acceptablelib_inds) < num_libs:
            num_libs = len(lib_opts.acceptablelib_inds) - 1
        lib_inds = np.linspace(1, len(lib_opts.acceptablelib_inds) - 1, num_libs, dtype=int)

    if skip_0 is True:
        lib_inds = lib_inds[1:]

    if lags is None:
        lag_amplitude = max([Tp_max, 3])
        lag_spacing = max([1, .1 * lag_amplitude])
        lags = np.concatenate(
            [np.arange(0, 2 * max([Tp_max, 3]) + 1, lag_spacing), np.arange(-2 * max([Tp_max, 3]), 0, lag_spacing)],
            axis=0)
        np.sort(lags)

    if train_len is None:
        train_len = (len(df_raw) - lib_opts.acceptablelib_inds[-1]) - npartial

    for iq, lib_sub_ind in enumerate(lib_inds):

        train_ind_i = lib_opts.acceptablelib_inds[lib_sub_ind]
        train_ind_f = min([train_ind_i + train_len + npartial, len(df_raw)])
        df_sub = df_raw.iloc[train_ind_i:train_ind_f].copy().reset_index()
        # print(df_sub.columns)
        # df_sub= df_sub.drop(columns=['index'])
        if iq % 10 == 0:
            # print('lib_sub_ind', lib_sub_ind)
            print(iq, 'train_ind_i:train_ind_f', train_ind_i, train_ind_f)

        for lag in lags:
            shifted = df_sub.copy()
            shifted[target_var] = shifted[target_var].shift(lag)
            shifted = shifted.dropna()

            max_liblen = len(shifted) - npartial - Tp_max

            if libSizes is not None:
                if not isinstance(libSizes, np.ndarray):
                    libSizes = np.array(libSizes)
                libSizes = libSizes[libSizes >= E + 1]
                libSizes = libSizes[libSizes <= min([max_liblen, train_len])]
                libSizes = list(libSizes)
            else:
                if num_libSizes is None:
                    if min([max_liblen, train_len]) - max([2, E + 1]) < 40:
                        libSizes = list(np.arange(max([2, E + 1]), min([max_liblen, train_len]), 1))
                    else:
                        libSizes = list(np.linspace(max([2, E + 1]), min([max_liblen, train_len]), 40, dtype=int))
                else:
                    if min([max_liblen, train_len]) - max([2, E + 1]) > num_libSizes:
                        libSizes = list(
                            np.linspace(max([2, E + 1]), min([max_liblen, train_len]), num_libSizes, dtype=int))
                    else:
                        libSizes = list(np.arange(max([2, E + 1]), min([max_liblen, train_len]), 1))

            ccm_out = pe.CCM(dataFrame=shifted,
                             E=E, Tp=Tp_max, tau=tau, exclusionRadius=0, knn=knn, verbose=False,
                             columns=col_var, target=target_var, libSizes=libSizes,
                             sample=sample, embedded=False, seed=seed, includeData=True, returnObject=True,
                             showPlot=False, noTime=False)

            ccm_out_df = pd.concat(
                [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
            ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, train_ind_i, train_ind_f, lag=lag)

            ccm_stash.append(ccm_out_df)
            if isinstance(ccm_d, list):
                ccm_d.append(ccm_out_df)
            else:
                ccm_d[(E, tau)].append(ccm_out_df)
    return ccm_d#ccm_stash