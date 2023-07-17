import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_1samp, ttest_ind

class LDA(object):
    def __init__(
        self
        ):
        self.mean_scores = []
        self.std_scores = []
        self.dvals = []
        self.t_stats = []
        self.p_vals = []
        self.low_bet_avg_powers = []
        self.high_bet_avg_powers = []
        self.diff_avg_powers = []
        self.lda_coefs = []

    def __create_X(self, data, channel, time):
        if type(time) == int:
            X = data[:, channel, :, time:time+self._time_resolution].mean(-1)
        else:
            X = data[:, channel, :, time]

        return X
    
    def __reshape_attributes(self, new_shape:tuple):
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, np.reshape(getattr(self, attr_name), new_shape))

    def __set_attributes(self, data, time_resolution):
        self._num_trials, self._num_channels, self._num_freqs, self._num_timesteps = data.shape

        # Checks that input value for time resolution is valid
        if time_resolution == None:
            pass
        elif not(self._num_timesteps % time_resolution == 0):
            raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
        else:
            self._time_resolution = time_resolution
            self._rescaled_timesteps = int(self._num_timesteps/time_resolution)


    def __train(self, X, y):
        low_bet_avg_powers = X[np.where(y == 0)].mean(0)
        high_bet_avg_powers = X[np.where(y == 1)].mean(0)

        self.high_bet_avg_powers.append(high_bet_avg_powers)
        self.low_bet_avg_powers.append(low_bet_avg_powers)
        self.diff_avg_powers.append(high_bet_avg_powers - low_bet_avg_powers)

        # Using RepeatedKFold() for training LDA
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

        estimators = []
        scores = []
        dval = np.zeros(self._num_trials)

        for train, test in rkf.split(X):
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X[train], y[train])
            estimators.append(lda)
            scores.append(lda.score(X[test], y[test]))
            dval[test] = np.dot(X[test], lda.coef_.T).T[0] + lda.intercept_

        self.dvals.append(dval)
        self.lda_coefs.append(estimators[scores.index(max(scores))].coef_[0])
        self.mean_scores.append(np.mean(scores))
        
        self.std_scores.append(np.std(scores))

        t_stat, p_val = ttest_1samp(dval, popmean=0) # perform 1-sided t-test on decision values corresponding to high bet
        self.t_stats.append(t_stat)
        self.p_vals.append(p_val)

    def train_per_channel_and_timestep(self, data, y, time_resolution):
        self.__set_attributes(data, time_resolution)

        for channel in range(self._num_channels):
            for time in range(self._rescaled_timesteps):
                X = self.__create_X(data, channel, time*self._time_resolution)
                self.__train(X, y)

        self.__reshape_attributes((self._num_channels,self._rescaled_timesteps,-1))

    def train_on_all_channels(self, data, y, time_resolution):
        self.__set_attributes(data, time_resolution)

        for time in range(self._rescaled_timesteps):
            X = self.__create_X(data, np.arange(self._num_channels), time*self._time_resolution)
            X_reshaped = X.reshape(self._num_trials,-1)
            self.__train(X_reshaped,y)
        
        self.__reshape_attributes((self._rescaled_timesteps,-1))
    
    def train_on_all_timesteps(self, data, y):
        self.__set_attributes(data, None)

        for channel in range(self._num_channels):
            X = self.__create_X(data, channel, np.arange(self._num_timesteps))
            X_reshaped = X.reshape(self._num_trials, -1)
            self.__train(X_reshaped, y)
        
        self.__reshape_attributes((self._num_channels,-1))