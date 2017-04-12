#!/usr/bin/python
# coding: utf-8
import sklearn.cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import utils
# import custom_hmm


class ClassificationHelper(object):
    def __init__(self, classifier_name, classes=[0, 1], classifier=None, seed=1234):
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = self._build_classifier_from(classifier_name, seed=seed)
            print "Using {} classifier".format(self.classifier)

        self.classifier_name = classifier_name
        self.classes = classes

    def _build_classifier_from(self, classifier_name, seed=1234):
        classifiers = {"SVM": SVC(kernel="linear", probability=True, C=1, class_weight='balanced'),  # Balanced da pesos según cantidad de instancias
                       "SVMG": SVC(kernel='rbf', probability=True, gamma=0.7, C=1, class_weight='balanced'),
                       "RF10": RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=-1, random_state=seed, class_weight='balanced'),
                       "RF100": RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=seed, class_weight='balanced'),
                       "RF1000": RandomForestClassifier(n_estimators=1000, criterion='gini', n_jobs=-1, random_state=seed, class_weight='balanced'),
                       #    "HMM": custom_hmm.ClassificationHMM(),
                       }

        return classifiers[classifier_name]

    def classification_probas(self, X_train, y_train, X_test, seed=1234):
        self.classifier.fit(X_train, y_train)
        return self.classifier.predict_proba(X_test)

    def one_speaker_out_cross_validation(self, X, y, speakers, save_weights, verbose=True, seed=1234):
        folds = sklearn.cross_validation.LeaveOneLabelOut(speakers)
        results = {}
        results[self.classifier_name] = {}

        for i, (train_index, test_index) in enumerate(folds):
            if verbose:
                utils.print_inline("speaker: {} of {} for {}".format(i + 1, len(set(speakers)), self.classifier_name))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            results[self.classifier_name][i] = self._classify(X_train, X_test, y_train, save_weights)
            results[self.classifier_name][i]["actual"] = y_test
            results[self.classifier_name][i]["y_ids"] = test_index

        return {"categories": self.classes, "results": results, "y": y}

    def k_fold_cross_validation(self, X, y, n_folds, save_weights, verbose=True, seed=1234, subsample=False):
        counts = [sum(y == c) for c in self.classes]

        if verbose:
            print "running classifiers for", zip(self.classes, counts)
        folds = sklearn.cross_validation.StratifiedKFold(y, n_folds=n_folds, random_state=seed)

        results = {}
        results[self.classifier_name] = {}

        for i, (train_index, test_index) in enumerate(folds):
            if verbose:
                utils.print_inline("K-fold: {} of {} for {}".format(i + 1, n_folds, self.classifier_name))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if subsample:
                sub_ids = utils.subsample_ids(y_train)
                X_train = X_train[sub_ids]
                y_train = y_train[sub_ids]

            results[self.classifier_name][i] = self._classify(X_train, X_test, y_train, save_weights)
            results[self.classifier_name][i]["actual"] = y_test

        return {"categories": self.classes, "results": results, "y": y}

    def _classify(self, X_train, X_test, y_train, save_weights):
        self.classifier.fit(X_train, y_train)
        y_pred_probabilities = self.classifier.predict_proba(X_test)
        res = {"predicted_probabilities": y_pred_probabilities}

        if save_weights:
            try:
                feature_importances = self.classifier.feature_importances_
            except:
                try:
                    feature_importances = self.classifier.coef_
                except:
                    feature_importances = None
            res.update({"classifier_weights": feature_importances})
        return res
