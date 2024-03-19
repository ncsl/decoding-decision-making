import numpy as np
from estimator import Estimator, EstimatorTrainOptimalTimeWindows

from sklearn import ensemble

class RandomForest(Estimator):
    def train(self, X, y):
        low_bet_powers = X[np.where(y == 0)]
        high_bet_powers = X[np.where(y == 1)]
        diff_avg_powers = high_bet_powers.mean(0) - low_bet_powers.mean(0)

        self.high_bet_powers.append(high_bet_powers)
        self.low_bet_powers.append(low_bet_powers)
        self.diff_avg_powers.append(diff_avg_powers)

        clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, oob_score=True)
        clf.fit(X, y)

        dval = clf.predict_proba(X)[:, 1] - 0.5
        dval = np.asarray(dval)

        self.mean_scores.append(clf.oob_score_)
        self.dvals.append(dval)

class RandomForestOptimal(RandomForest, EstimatorTrainOptimalTimeWindows):
    def __init__(self, data, setup_data):
        super().__init__(data, setup_data)