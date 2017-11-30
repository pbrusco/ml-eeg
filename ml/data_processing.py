# coding=utf-8

import numpy as np


def handle_nans(nan_action, X, features_names):
    nan_rows = np.isnan(X).any(axis=1)

    if nan_action == "none" and sum(nan_rows) != 0:
        raise Exception("Error, data contains {} of {} rows with NaN, set the nan_action".format(sum(nan_rows), len(nan_rows)))

    if nan_action == "remove":
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows, :]
        print("FILTERNING NaN rows")
        deleted_rows = nan_rows

    if nan_action in ["replace_by_zero", "replace_by_mean", "replace_by_median"]:
        X = np.hstack([X, np.isnan(X) * 1.0])
        features_names = features_names + ["was_nan" + f for f in features_names]
        deleted_rows = np.zeros(X.shape[0], np.bool)

    if nan_action == "replace_by_zero":
        print(("converting {} NaN to 0s".format(np.count_nonzero(~np.isnan(X)))))
        X = np.nan_to_num(X)
        deleted_rows = np.zeros(X.shape[0], np.bool)

    if nan_action == "replace_by_mean":
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        deleted_rows = np.zeros(X.shape[0], np.bool)

    if nan_action == "replace_by_median":
        col_mean = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        deleted_rows = np.zeros(X.shape[0], np.bool)

    return X, deleted_rows, features_names
