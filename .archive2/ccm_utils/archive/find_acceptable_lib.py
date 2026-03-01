import numpy as np


class AcceptableLib:
    def __init__(self, la, acceptablelib, lengthacceptablelib, A_start_ind=0):
        beg_inds = la - acceptablelib
        self.acceptablelib_inds = beg_inds[::-1] + (A_start_ind - 1)
        self.maxliblen = lengthacceptablelib


def find_acceptable_libstart_maxlen(lA: object, E: object, tau: object, Tp: object = 1, l_max: object = 0, A_start_ind: object = 0) -> object:
    """
    Find the acceptable library start indexes and the maximum library length
    :param lA: 
    :param E: 
    :param tau: 
    :param Tp: 
    :param l_max: 
    :param A_start_ind: 
    :return: 
    """

    # if l_max is set to something other than 0,
    # the indexes will support a minimum maxliblen
    A = np.arange(0, lA)
    # A = A[A_start_ind:]
    gapdist = np.abs(tau) * (E - 1) + l_max + Tp
    acceptablelib = np.isfinite(A).astype(int)
    # lA = len(A)

    for i in range(1, gapdist):
        # print('i', i)
        # assumes standard delay direction
        # Create a new array by appending `i` `np.nan` values to the front of the array `A` excluding the last `i` elements
        new_A = np.concatenate([np.full(i, np.nan), A[:-i]])
        # print(new_A)

        # I think this would handle the other direction:
        # new_A = np.concatenate([A[:-i],np.full(i, np.nan)])

        # Check if the elements in this new array are finite and convert the boolean results to integers
        finite_new_A = np.isfinite(new_A).astype(int)
        # print(finite_new_A)

        # Multiply the `acceptablelib` array by this integer array
        acceptablelib *= finite_new_A
        # acceptablelib<-acceptablelib*as.numeric(is.finite(c(rep(NA, i),A[-c((lA-i+1):lA)])))

    acceptablelib = np.where(acceptablelib > 0)[0] - 1  # hopefully C compatible
    lengthacceptablelib = len(acceptablelib)

    acceptablelib = AcceptableLib(lA, acceptablelib, lengthacceptablelib, A_start_ind=A_start_ind)
    return acceptablelib


def validate_parameters(E: object, Tp: object, tau: object, libPairs: object) -> object:
    # Calculate vectorStart and vectorEnd
    vectorStart = max((E - 1) * tau, 0)
    vectorStart = max(vectorStart, Tp)

    vectorEnd = min((E - 1) * tau, Tp)
    vectorEnd = min(vectorEnd, 0)

    # Calculate vectorLength
    vectorLength = abs(vectorStart - vectorEnd) + 1

    # Calculate maxLibrarySegment
    maxLibrarySegment = 0
    for pair in libPairs:
        libPairSpan = pair[1] - pair[0] + 1
        maxLibrarySegment = max(maxLibrarySegment, libPairSpan)

    # Print intermediate values
    print(f"vectorStart: {vectorStart}")
    print(f"vectorEnd: {vectorEnd}")
    print(f"vectorLength: {vectorLength}")
    print(f"maxLibrarySegment: {maxLibrarySegment}")

    # Perform validation check
    if vectorLength > maxLibrarySegment:
        return 0
    else:
        # print(f"Combination of E = {E}, Tp = {Tp}, tau = {tau} is valid.")
        return maxLibrarySegment


def make_test_set(lA: object, train_inds: object, E_max: object, tau_max: object, Tp_max: object) -> object:
    test_opts = []
    train_ind_i, train_ind_f = train_inds
    try:
        A = np.arange(0, train_ind_i)
        pre = find_acceptable_libstart_maxlen(
            len(A), E_max, tau_max, A_start_ind=0, Tp=Tp_max).acceptablelib_inds
        # pre[pre+ np.abs(tau_max) * (E_max - 1)<train_ind_i]
        test_opts.append(pre[pre + np.abs(tau_max) * (E_max - 1) < train_ind_i])
        # print(pre)
    except:
        pass
    try:
        A = np.arange(train_ind_f, lA)
        post = find_acceptable_libstart_maxlen(
            len(A), E_max, tau_max, A_start_ind=train_ind_f, Tp=Tp_max).acceptablelib_inds
        test_opts.append(post[post > train_ind_f])  # + np.abs(tau_max) * (E_max - 1)<train_ind_i])
        # print(post)
    except:
        pass

    test_inds = []
    if len(test_opts) > 0:
        for ip in range(len(test_opts)):
            if len(test_opts[ip]) > 0:
                # if test_opts[ip][0]<train_ind_i & test_opts[ip][0] + np.abs(tau_max) * (E_max - 1) < train_ind_i:
                #     test_start = test_opts[ip][0]
                #     test_end = test_opts[ip][test_opts[ip]+np.abs(tau_max) * (E_max - 1)<train_ind_i][-1]
                #
                #
                # elif test_opts[ip][0]>train_ind_f:
                #     test_start = test_opts[ip][0]
                #     & test_opts[ip][-1] + np.abs(tau_max) * (
                #         E_max - 1) > train_ind_f):
                test_inds.append([test_opts[ip][0],
                                  max([test_opts[ip][-1], test_opts[ip][0] + np.abs(tau_max) * (E_max - 1)])])

    return test_inds
