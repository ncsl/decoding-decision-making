import numpy as np
from sklearn.model_selection import RepeatedKFold
from estimator import Estimator, EstimatorTrainOptimalTimeWindows

from sklearn import linear_model

class LogisticRegression(Estimator):
    """Class to train Logistic Regression model on data"""
    def train(self, X, y):
        low_bet_powers = X[np.where(y == 0)]
        high_bet_powers = X[np.where(y == 1)]
        diff_avg_powers = high_bet_powers.mean(0) - low_bet_powers.mean(0)

        self.high_bet_powers.append(high_bet_powers)
        self.low_bet_powers.append(low_bet_powers)
        self.diff_avg_powers.append(diff_avg_powers)

        # Using RepeatedKFold() for training Logistic Regression model
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

        estimators = []
        scores = []
        dval = np.zeros(self._num_trials)

        for train, test in rkf.split(X):
            clf = linear_model.LogisticRegression(penalty='l2', solver='lbfgs', C=1e-1).fit(X[train], y[train])
            estimators.append(clf)
            scores.append(clf.score(X[test], y[test]))
            dval[test] = np.dot(X[test], clf.coef_.T).T[0] + clf.intercept_
            
        self.mean_scores.append(np.mean(scores))
        self.dvals.append(dval)

class LogisticRegressionOptimal(LogisticRegression, EstimatorTrainOptimalTimeWindows):
    def __init__(self, data, setup_data):
        super().__init__(data, setup_data)