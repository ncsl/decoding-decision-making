import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_1samp

class LDA(object):
    def __init__(
        self,
        data = None,
        setup_data = None
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

        self.__set_attributes(data, setup_data)

    @property
    def num_trials(self):
        return self.__num_trials

    @property
    def num_channels(self):
        return self.__num_channels

    @property
    def num_freqs(self):
        return self.__num_freqs
    
    @property
    def num_timesteps(self):
        return self.__num_timesteps
    
    @property
    def time_resolution(self):
        return self.__time_resolution
    
    @property
    def rescaled_timesteps(self):
        return self.__rescaled_timesteps

    def __create_X(self, data, channel, time):
        if type(time) == int:
            X = data[:, channel, :, time:time+self.__time_resolution].mean(-1)
        else:
            X = data[:, channel, :, time]

        return X
    
    def __reshape_attributes(self, new_shape:tuple):
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, np.reshape(getattr(self, attr_name), new_shape))

    def __set_attributes(self, data, setup_data):
        self.__num_trials, self.__num_channels, self.__num_freqs, self.__num_timesteps = data.shape

        self.__elec_areas = setup_data['elec_area']
        self.__elec_names = setup_data['elec_name']

    def __filter_channels(self):
        return [i for i,ea in enumerate(self.__elec_areas) if ea not in ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid','lesion L','ventricle L','ventricle R']]

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
        dval = np.zeros(self.__num_trials)

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

    def create_cluster_idxs(self, threshold):
        if self.t_stats.shape == (self.__num_channels, self.__rescaled_timesteps, 1):
            cluster_idxs = []

            for channel in range(self.__num_channels):
                ch_cluster_idxs = []
                threshold_idxs = [i for i, t_stat in enumerate(self.t_stats[channel].flatten()) if np.abs(t_stat) > threshold]
                temp_cluster_idxs = [threshold_idxs[0]]
            
                # Groups consecutive clusters together
                for i in range(len(threshold_idxs) - 1):
                    if threshold_idxs[i+1] - threshold_idxs[i] == 1:
                        temp_cluster_idxs.append(threshold_idxs[i+1])
                    else:
                        ch_cluster_idxs.append(temp_cluster_idxs)
                        temp_cluster_idxs = [threshold_idxs[i+1]]
                
                if len(temp_cluster_idxs) != 0:
                    ch_cluster_idxs.append(temp_cluster_idxs)
                
                cluster_idxs.append(ch_cluster_idxs)
            
            return cluster_idxs
        else:
            Exception('Cannot create clusters with these attributes, make sure shape of attributes is (num_channels, rescaled_timesteps, 1)')

    def compute_t_stat_clusters(self, threshold):
        t_stat_sums = []

        for channel in range(self.num_channels):
            temp_t_stat_sums = []
            for arr in self.create_cluster_idxs(threshold)[channel]:
                temp_t_stat_sums.append(self.t_stats[channel][arr].sum())
            
            t_stat_sums.append(temp_t_stat_sums)
        
        return t_stat_sums

    def train_per_channel_and_timestep(self, data, y, time_resolution):
        if not(self.__num_timesteps % time_resolution == 0):
            raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
        else:
            self.__time_resolution = time_resolution
            self.__rescaled_timesteps = int(self.__num_timesteps/time_resolution)

        for channel in range(self.__num_channels):
            for time in range(self.__rescaled_timesteps):
                X = self.__create_X(data, channel, time*self.__time_resolution)
                self.__train(X, y)

        self.__reshape_attributes((self.__num_channels,self.__rescaled_timesteps,-1))

    def train_on_all_channels(self, data, y, time_resolution, filter_channels:bool = False):
        if not(self.__num_timesteps % time_resolution == 0):
            raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
        else:
            self.__time_resolution = time_resolution
            self.__rescaled_timesteps = int(self.__num_timesteps/time_resolution)

        for time in range(self.__rescaled_timesteps):
            if filter_channels:
                X = self.__create_X(data, self.__filter_channels(), time*self.__time_resolution)
            else:
                X = self.__create_X(data, np.arange(0,self.__num_channels), time*self.__time_resolution)

            X_reshaped = X.reshape(self.__num_trials,-1)
            self.__train(X_reshaped,y)
        
        self.__reshape_attributes((self.__rescaled_timesteps,-1))
    
    def train_on_all_timesteps(self, data, y):
        for channel in range(self.__num_channels):
            X = self.__create_X(data, channel, np.arange(self.__num_timesteps))
            X_reshaped = X.reshape(self.__num_trials, -1)
            self.__train(X_reshaped, y)
        
        self.__reshape_attributes((self.__num_channels,-1))

class ShuffledLDA(LDA):
    def __init__(self, 
                 data=None, 
                 setup_data=None):
        super().__init__(data, setup_data)
        self.__get_hand(setup_data)

        
    def __get_hand(self, setup_data):
        bets = setup_data['filters']['bets']
        good_trials = np.where(np.isnan(bets) == False)[0] # extract indices of trials without the 'nan'

        self.__sub_hand = setup_data['filters']['card1'][good_trials] # get the subject's card hand for the good trials
    
    def __shuffle_y(self, y):
        """
        Randomly shuffle the elements of y to be in different locations.

        Parameters
        ----------
            y : arr, required
        The labels that the LDA model is to be trained on

        Returns
        -------
            y_shuffled : arr
        An array with the elements of y randomly shuffled to be in different locations.
        """
        
        np.random.seed()

        print('Shuffling!')

        # Get the locations for each particular card value
        card_value_indices = []
        for i in [2,4,6,8,10]:
            card_value_indices.append(np.where(self.__sub_hand == i)[0])

        y_shuffled = np.zeros(y.shape)

        # Ensure that the number of high bets in the shuffled y labels is consistent with the card value
        for indices in card_value_indices:
            temp = indices
            num_high_bets = y[indices].sum() + round(np.random.uniform(-1,1)*y[indices].sum()*0.2) # Get the number of high bets for a particular card value and add some randomness to it
            for _ in range(num_high_bets):
                if np.any(temp):
                    # Pick a random location from all possible locations of that particular card value and set it to 1 (ie high bet)
                    rand = np.random.choice(temp)
                    y_shuffled[rand] = 1
                    rand_index = np.where(temp == rand)[0]
                    temp = np.delete(temp,rand_index) # Remove that location from being able to be chosen again
            y_shuffled[temp] = 0 # set all other locations for that particular card value to 0 (ie low bet)

        return y_shuffled
    
    def train_per_channel_and_timestep(self, data, y, time_resolution):
        y_shuffled = self.__shuffle_y(y)
        super().train_per_channel_and_timestep(data, y_shuffled, time_resolution)

    def compute_t_stat_clusters(self, cluster_estimator, threshold):
        t_stat_sums = []

        for channel in range(self.num_channels):
            temp_t_stat_sums = []
            for arr in cluster_estimator.create_cluster_idxs(threshold)[channel]:
                temp_t_stat_sums.append(self.t_stats[channel][arr].sum())
            
            t_stat_sums.append(temp_t_stat_sums)
        
        return t_stat_sums