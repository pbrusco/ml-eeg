#!/usr/bin/python
# coding: utf-8

from hmmlearn import hmm
import numpy as np
import math

N_CHANNELS = 128


class ClassificationHMM:
    def __init__(self, hmm_builder=None):
        if not hmm_builder:
            def hmm_builder():
                model = hmm.GMMHMM(n_components=3, n_mix=1, covariance_type="diag", init_params="t")
                model.transmat_ = np.array([[0.5, 0.5, 0.0],
                                            [0.0, 0.5, 0.5],
                                            [0.0, 0.0, 1.0]])
                return model

        self.hmms_0 = [hmm_builder() for i in range(0, N_CHANNELS)]
        self.hmms_1 = [hmm_builder() for i in range(0, N_CHANNELS)]
        self.feature_importances_ = None

    def fit(self, X_train, y_train):
        labels = set(y_train)
        if len(labels) != 2:
            raise Exception("y_train doesn't contain 2 classes")
        X_0 = X_train[y_train == 0, :, :]
        X_1 = X_train[y_train == 1, :, :]

        for ch in range(0, N_CHANNELS):
            self.hmms_0[ch].fit(X_0[:, ch, :])
            self.hmms_1[ch].fit(X_1[:, ch, :])

    def predict_proba(self, X_test):
        probas = []
        for x_test in X_test:
            hmm_probas_by_channel = []
            for ch in range(0, N_CHANNELS):
                hmm0 = self.hmms_0[ch]
                hmm1 = self.hmms_1[ch]
                x = x_test[ch, :]
                log_prob_hmm0 = hmm0.score([x])
                log_prob_hmm1 = hmm1.score([x])

                prob_x_hmm0 = self.probas_from_score(log_prob_hmm0, log_prob_hmm1)
                prob_x_hmm1 = self.probas_from_score(log_prob_hmm1, log_prob_hmm0)

                assert round(prob_x_hmm0 + prob_x_hmm1, 1) == 0, "probs doen't sum 1.\t P(x|hmm0) + P(x|hmm1) = {}".format(prob_x_hmm0 + prob_x_hmm1)
                assert prob_x_hmm0 <= 1 and prob_x_hmm0 >= 0
                assert prob_x_hmm1 <= 1 and prob_x_hmm1 >= 0
                hmm_probas_by_channel.append([prob_x_hmm0, prob_x_hmm1])

        max_proba = 0
        for ch, (prob_hmm0, prob_hmm1) in enumerate(hmm_probas_by_channel):
            m = max(max_proba, prob_hmm0, prob_hmm1)
            if m > max_proba:
                max_proba = m
                prob_x = (prob_hmm0, prob_hmm1)
            probas.append(prob_x)

        return np.array(probas)

    def probas_from_score(self, lp1, lp2):
        try:
            # total = logsumexp([lp0, lp1])
            # return np.exp(lp0 - total)
            div = math.exp(lp2 - lp1)
            return (div + 1) ** (-1)
        except OverflowError:
            return 0.0
