# coding: utf-8

import sklearn.cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from . import utils


class ClassificationHelper(object):
    def __init__(self, config, classes=[0, 1], classifier=None):
        self.config = config

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = self._build_classifier()
            print("Using {} classifier".format(self.classifier))

        self.classes = classes

    def _build_classifier(self):
        classifiers = {"SVM": SVC(kernel="linear", probability=True, C=1, class_weight='balanced'),  # Balanced da pesos según cantidad de instancias
                       "SVMG": SVC(kernel='rbf', probability=True, gamma=0.7, C=1, class_weight='balanced'),
                       "RF": RandomForestClassifier(n_estimators=self.config["classifier_n_trees"], criterion='gini', n_jobs=-1, random_state=self.config["seed"], class_weight='balanced'),
                       }

        return classifiers[self.config["classifier_name"]]

    def classification_probas(self, X_train, y_train, X_test):
        self.classifier.fit(X_train, y_train)
        return self.classifier.predict_proba(X_test)

    def one_speaker_out_cross_validation(self, X, y, speakers, save_weights, verbose=True):
        folds = sklearn.cross_validation.LeaveOneLabelOut(speakers)
        results = {}
        results = {}

        for i, (train_index, test_index) in enumerate(folds):
            if verbose:
                utils.print_inline("speaker: {} of {} for {}".format(i + 1, len(set(speakers)), self.config["classifier_name"]))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            results[i] = self._classify(X_train, X_test, y_train, save_weights)
            results[i]["actual"] = y_test
            results[i]["y_ids"] = test_index

        return {"categories": self.classes, "results": results, "y": y}

    def k_fold_cross_validation(self, X, y, n_folds, save_weights, verbose=True, subsample=False):
        counts = [sum(y == c) for c in self.classes]

        if verbose:
            print("running classifiers for", list(zip(self.classes, counts)))
        folds = sklearn.cross_validation.StratifiedKFold(y, n_folds=n_folds, random_state=self.config["seed"])

        fold_results = {}

        for i, (train_index, test_index) in enumerate(folds):
            if verbose:
                utils.print_inline("K-fold: {} of {} for {}".format(i + 1, n_folds, self.config["classifier_name"]))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if subsample:
                sub_ids = utils.subsample_ids(y_train)
                X_train = X_train[sub_ids]
                y_train = y_train[sub_ids]

            fold_results[i] = self._classify(X_train, X_test, y_train, save_weights)
            fold_results[i]["actual"] = y_test

        return {"categories": self.classes, "fold_results": fold_results, "y": y}

    def _classify(self, X_train, X_test, y_train, save_weights):
        self.classifier.fit(X_train, y_train)
        y_pred_probabilities = self.classifier.predict_proba(X_test)
        res = {"predicted_probabilities": y_pred_probabilities}

        if save_weights:
            if self.config["classifier_name"] == "RF":
                feature_importances = self.classifier.feature_importances_
            elif "SVM" in self.config["classifier_name"]:
                feature_importances = self.classifier.coef_
            else:
                raise "I don't know whitch weights to save"
            res.update({"classifier_weights": feature_importances})
        return res
