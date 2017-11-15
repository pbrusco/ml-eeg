# coding: utf-8


import numpy as np
from sklearn import metrics
from . import utils
import pandas


def calculate_measures(results, measures):
    processed_result = {}
    supports = []
    for (measure_name, measure_function) in measures:

        measure_result, pvalue, support = apply_measure(results, measure_function)
        processed_result[measure_name] = measure_result
        processed_result[measure_name + "_p_val"] = pvalue
        supports.append(support)

    assert(utils.all_equal(supports))

    support = supports[0]
    processed_result.update(dict(support=support))

    return processed_result


def apply_measure(results, measure_function):
    measure_result, support = calculate_result_for_measure(results, measure_function)

    if "permutations" in results and len(list(results["permutations"].keys())) > 0:
        permutation_values = [calculate_result_for_measure(perm_res, measure_function)[0] for perm_id, perm_res in results["permutations"].items()]
        pvalue = len([p for p in permutation_values if p >= measure_result]) * 1.0 / (len(permutation_values))
    else:
        pvalue = np.nan

    return measure_result, pvalue, support


def auc_fn(actual, predicted_probabilities, categories):
    return metrics.roc_auc_score(actual, predicted_probabilities)


def accuracy_fn(actual, predicted_probabilities, categories):
    predicted = [1 if p > 0.5 else 0 for p in predicted_probabilities]
    return metrics.accuracy_score(actual, predicted)


def y_indices_from_fold_result(fold_results):
    return np.array(utils.flatten([fold_res["y_ids"] for k, fold_res in fold_results.items()]))


def y_true_from_fold_result(fold_results):
    return np.array(utils.flatten([fold_res["actual"] for k, fold_res in fold_results.items()]))


def y_score_from_fold_result(fold_results):
    return np.array(utils.flatten([fold_res["predicted_probabilities"][:, 1] for k, fold_res in fold_results.items()]))


def calculate_result_for_measure(results, measure):
    categories = results["categories"]
    fold_results = results["fold_results"]

    actual = y_true_from_fold_result(fold_results)
    predicted_probabilities = y_score_from_fold_result(fold_results)

    measure_result = measure(actual, predicted_probabilities, categories)
    support = [sum(actual == c) for c in categories]

    return measure_result, list(zip(categories, support))


def feature_importances(fold_results):
    all_feature_importances = []

    for fold, fold_result in fold_results.items():
        all_feature_importances.append(fold_result["classifier_weights"])

    feature_importances_by_folds = np.array(all_feature_importances)
    return feature_importances_by_folds.mean(axis=0), feature_importances_by_folds.std(axis=0)


def feature_importances_table(fold_results, feature_mapping, session_id, subject):
    importance_means, importance_stds = feature_importances(fold_results)
    importances_res = []

    for feature_id, (importance_mean, importance_std) in enumerate(zip(importance_means, importance_stds)):
        info = feature_mapping[feature_id].copy()
        info["feature_importances_folds_mean"] = importance_mean
        info["feature_importances_folds_std"] = importance_std
        info["session"] = session_id
        info["subject"] = subject
        importances_res.append(info)

    features_table = pandas.DataFrame(importances_res)
    return features_table
