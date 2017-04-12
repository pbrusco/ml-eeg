#!/usr/bin/python
# coding: utf-8
import pickle
import system
import numpy as np
from sklearn import metrics
from pandas import DataFrame


def classification_results_for(session_id, subject, results_folder, experiment):
    filename = results_folder + "/{}_s{}_{}.p".format(experiment, session_id, subject)
    if system.exists(filename):
        return pickle.load(open(filename, "rb"))
    return None


def analize(results, experiment, measures):
    result = results[experiment]
    processed_result = {}
    for (measure_name, measure_function) in measures:
        measure_result, support = results_extractor(result, measure_function, permutation=False)
        processed_result[measure_name] = measure_result

        permutation_keys = [k for k in results.keys() if k.endswith("_permutation")]
        if len(permutation_keys) > 1:
            perm_results = [results_extractor(results[p], measure_function, permutation=True)[0] for p in permutation_keys]
            # processed_result[measure_name + "_perc_05"] = np.percentile(perm_results, 05)
            # processed_result[measure_name + "_perc_50"] = np.percentile(perm_results, 50)
            # processed_result[measure_name + "_perc_95"] = np.percentile(perm_results, 95)
            processed_result[measure_name + "_p_val"] = len([p for p in perm_results if p >= measure_result]) * 1.0 / (len(perm_results))

    total_support = support
    processed_result.update({
                            "experiment": experiment,
                            "support": total_support,
                            })
    return processed_result


def compare_subjects(sessions, results_folder, measures, experiments, warnings=True):
    res = []
    for (session_id, subject) in sessions:
        try:
            subject_session = (session_id, subject)
            for experiment in experiments:
                results = classification_results_for(session_id, subject, results_folder, experiment)
                if not results:
                    print "no results found for", (session_id, subject, results_folder, experiment)
                    continue
                row = analize(results, experiment, measures)
                row.update({"session": subject_session})
                res.append(row)
        except:
            continue

    return DataFrame(res)


def auc_fn(actual, predicted_probabilities, categories):
    predicted_proba = [p[1] for p in predicted_probabilities]
    return metrics.roc_auc_score(actual, predicted_proba)


def accuracy_fn(actual, predicted_probabilities, categories):
    predicted_proba = [p[1] for p in predicted_probabilities]
    predicted = [1 if p > 0.5 else 0 for p in predicted_proba]
    return metrics.accuracy_score(actual, predicted)


def results_extractor(results, measure, permutation):
    categories = results["categories"]
    classifier_results = results["results"][results["results"].keys()[0]]

    supports = []
    all_actuals = []
    all_predicted_probabilities = []
    all_feature_importances = []

    for i in classifier_results.keys():
        res_fold_i = classifier_results[i]
        actual = res_fold_i["actual"]
        predicted_probabilities = res_fold_i["predicted_probabilities"]
        all_actuals.extend(actual)

        if not permutation:
            all_feature_importances.append(res_fold_i["classifier_weights"])

        all_predicted_probabilities.extend(predicted_probabilities)
        supports.append([sum(actual == c) for c in categories])

    measure_result = measure(all_actuals, all_predicted_probabilities, categories)
    support = np.sum(supports, axis=0)

    return measure_result, zip(categories, support)


def feature_importances(results):
    classifier_results = results["results"][results["results"].keys()[0]]
    all_feature_importances = []

    for i in classifier_results.keys():
        res_fold_i = classifier_results[i]
        all_feature_importances.append(res_fold_i["classifier_weights"])

    feature_importances_by_folds = np.array(all_feature_importances)
    return feature_importances_by_folds.mean(axis=0), feature_importances_by_folds.std(axis=0)
