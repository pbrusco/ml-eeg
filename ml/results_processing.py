# coding: utf-8

from __future__ import division
import numpy as np
from sklearn import metrics


def analize(results, measures):
    processed_result = {}
    for (measure_name, measure_function) in measures:
        measure_result, support = results_extractor(results, measure_function, is_permutation=False)
        processed_result[measure_name] = measure_result

        for perm_id, perm_res in results["permutations"].iteritems():
            permutation_values = [results_extractor(perm_res, measure_function, is_permutation=True)[0] for perm_id, perm_res in results["permutations"].iteritems()]
            processed_result[measure_name + "_p_val"] = len([p for p in permutation_values if p >= measure_result]) * 1.0 / (len(permutation_values))

    total_support = support
    processed_result.update(dict(support=total_support))

    return processed_result


def auc_fn(actual, predicted_probabilities, categories):
    predicted_proba = [p[1] for p in predicted_probabilities]
    return metrics.roc_auc_score(actual, predicted_proba)


def accuracy_fn(actual, predicted_probabilities, categories):
    predicted_proba = [p[1] for p in predicted_probabilities]
    predicted = [1 if p > 0.5 else 0 for p in predicted_proba]
    return metrics.accuracy_score(actual, predicted)


def results_extractor(results, measure, is_permutation):
    categories = results["categories"]
    classifier_results = results["fold_results"]

    supports = []
    all_actuals = []
    all_predicted_probabilities = []

    for i in list(classifier_results.keys()):
        res_fold_i = classifier_results[i]
        actual = res_fold_i["actual"]
        predicted_probabilities = res_fold_i["predicted_probabilities"]
        all_actuals.extend(actual)

        all_predicted_probabilities.extend(predicted_probabilities)
        supports.append([sum(actual == c) for c in categories])

    measure_result = measure(all_actuals, all_predicted_probabilities, categories)
    support = np.sum(supports, axis=0)

    return measure_result, list(zip(categories, support))


def feature_importances(results):
    results = results["fold_results"]
    all_feature_importances = []

    for fold, fold_results in results.iteritems():
        all_feature_importances.append(fold_results["classifier_weights"])

    feature_importances_by_folds = np.array(all_feature_importances)
    return feature_importances_by_folds.mean(axis=0), feature_importances_by_folds.std(axis=0)
