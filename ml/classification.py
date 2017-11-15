# coding: utf-8

import sklearn.cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from . import utils


class ClassificationHelper(object):
    def __init__(self, config, classes=[0, 1], classifier=None):
        self.config = config
        self.subsample = config["subsample"]

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = self._build_classifier()
            print(("Using {} classifier".format(self.classifier)))

        self.classes = classes

    def _build_classifier(self):
        classifiers = {
            "SVM": SVC(),
            "RF": RandomForestClassifier(),
            "LDA": LinearDiscriminantAnalysis(),
        }
        clf = classifiers[self.config["classifier_name"]]
        clf.set_params(**self.config["classifier_params"])
        return clf

    def classification_probas(self, X_train, y_train, X_test):
        self.classifier.fit(X_train, y_train)
        return self.classifier.predict_proba(X_test)

    def one_speaker_out_cross_validation(self, X, y, speakers):
        folds = sklearn.cross_validation.LeaveOneLabelOut(speakers)
        return self._cross_validate(folds, X, y)

    def k_fold_cross_validation(self, X, y, n_folds, verbose=True):
        counts = [sum(y == c) for c in self.classes]
        if verbose:
            print(("running classifiers for", list(zip(self.classes, counts))))
        folds = sklearn.cross_validation.StratifiedKFold(y, n_folds=n_folds, random_state=self.config["seed"])
        return self._cross_validate(folds, X, y)

    def _cross_validate(self, folds, X, y):
        fold_results = {}
        n_folds = len(folds)

        for i, (train_index, test_index) in enumerate(folds):
            utils.print_inline("fold: {} of {}".format(i + 1, n_folds))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.subsample:
                sub_ids = utils.subsample_ids(y_train)
                X_train = X_train[sub_ids]
                y_train = y_train[sub_ids]

            fold_results[i] = self._classify(X_train, X_test, y_train)
            fold_results[i]["actual"] = y_test
            fold_results[i]["y_ids"] = test_index

        return {"categories": self.classes, "fold_results": fold_results, "y": y}

    def _classify(self, X_train, X_test, y_train):
        self.classifier.fit(X_train, y_train)
        y_pred_probabilities = self.classifier.predict_proba(X_test)
        res = {"predicted_probabilities": y_pred_probabilities}

        if self.config["save_weights"]:
            if self.config["classifier_name"] == "RF":
                feature_importances = self.classifier.feature_importances_
            elif "SVM" in self.config["classifier_name"]:
                feature_importances = self.classifier.coef_
            else:
                raise "I don't know whitch weights to save"
            res.update({"classifier_weights": feature_importances})
        return res
