import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_1samp

from multiprocessing import Pool

def _generate_sampled_channels(channels, sample_size, sampled_channels_=[]):
    """Recursively generate a list (size = sample_size) of lists of sampled channels"""
    np.random.seed()

    if len(channels) > sample_size:
        sample = np.random.choice(channels, size=sample_size, replace=False)
        sampled_channels_.append(list(sample))
        channels = np.delete(channels, np.where(np.isin(channels, sample))[0])
        _generate_sampled_channels(channels, sample_size, sampled_channels_)
    else:
        sampled_channels_.append(list(channels))

    return sampled_channels_

class LDA(object):
    def __init__(
            self,
            setup_data
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

        self.__elec_areas = setup_data['elec_area']
        self.__elec_names = setup_data['elec_name']

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
    
    @property
    def elec_areas(self):
        return self.__elec_areas
    
    @property
    def elec_names(self):
        return self.__elec_names
    
    @property
    def optimal_time_windows_per_channel(self):
        if self.__optimal_time_windows_per_channel == None:
            raise Exception('No optimal time windows have been computed yet')
        return self.__optimal_time_windows_per_channel

    def _create_time_windows(self):
        """Create all possible time windows from which X data can be created."""
        time_windows = []
        for i in range(self.__num_timesteps):
            for j in range(self.__num_timesteps):
                if i-j >= 0 and i+j <= self.__num_timesteps:
                    time_windows.append([i-j,i+j])
                else:
                    break
        
        return time_windows

    def _create_X(self, data, channel, time):
        """Create the X data that will be used to train the LDA model"""
        if type(time) == int:
            X = data[:, channel, :, time:time+self.__time_resolution].mean(-1)
        elif type(time) == list and len(time) == 2:
            if time[0] - time[1] == 0:
                X = data[:, channel, :, time[0]]
            else:
                X = data[:, channel, :, time[0]:time[1]].mean(-1)
        else:
            pass

        return X

    def _filter_channels(self):
        """Filters out channels that are in particular anatomical locations"""
        self.__filtered_elec_areas_idxs = [i for i,ea in enumerate(self.__elec_areas) if ea not in ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid','lesion L','ventricle L','ventricle R']]
        self.__filtered_elec_areas = [self.__elec_areas[i] for i in self.__filtered_elec_areas_idxs]

    def _multiprocessing_time_window_grid_search(self, data, y, n_processes, filter_channels:bool = True):
        """Perform a time window grid search in parallel"""
        self._set_attributes(data)

        if filter_channels:
            self._filter_channels()
            channels = self.__filtered_elec_areas_idxs
        else:
            channels = np.arange(self.__num_channels)

        sample_size = round(len(channels)/n_processes)

        sampled_channels = _generate_sampled_channels(channels, sample_size, [])

        if __name__ == '__main__':
            with Pool(n_processes) as p:
                results = p.starmap(self._time_window_grid_search, [(data, y, channels) for channels in sampled_channels])
                p.close()
        
        return results

    def _reshape_attributes(self, new_shape:tuple):
        """Reshape class attributes to specified shape"""
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, np.reshape(getattr(self, attr_name), new_shape))
        
    def _reset_attributes(self):
        self.mean_scores = []
        self.std_scores = []
        self.dvals = []
        self.t_stats = []
        self.p_vals = []
        self.low_bet_avg_powers = []
        self.high_bet_avg_powers = []
        self.diff_avg_powers = []
        self.lda_coefs = []

    def _set_attributes(self, data, **kwargs):
        """Set class attributes specified by the dataset and metadata"""
        self.__num_trials, self.__num_channels, self.__num_freqs, self.__num_timesteps = data.shape

        if 'time_resolution' in kwargs:
            if not(self.__num_timesteps % kwargs['time_resolution'] == 0):
                raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
            else:
                self.__time_resolution = kwargs['time_resolution']
                self.__rescaled_timesteps = int(self.__num_timesteps/kwargs['time_resolution'])
    
    def _time_window_grid_search(self, data, y, channels):
        """Train LDA model on all possible time windows, store the time windows that correspond with the highest LDA score."""
        self._set_attributes(data)
        time_windows = self._create_time_windows()
        best_time_windows = []

        for channel in channels:
            for times in time_windows:
                X = self._create_X(data, channel, times)
                self._train(X,y)

            print(f'Channel {channel} done')
            best_time_windows.append([channel, time_windows[np.argmax(self.mean_scores)], np.max(self.mean_scores)])
            self._reset_attributes()
        
        return best_time_windows

    def _train(self, X, y):
        """Train LDA model on specified X data and y labels"""
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
        """Creates statistical clusters based on the t-statistics computed from LDA decision values"""
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
        """Computes the sum of the t-statistics in each cluster"""
        t_stat_sums = []

        for channel in range(self.num_channels):
            temp_t_stat_sums = []
            for arr in self.create_cluster_idxs(threshold)[channel]:
                temp_t_stat_sums.append(self.t_stats[channel][arr].sum())
            
            t_stat_sums.append(temp_t_stat_sums)
        
        return t_stat_sums

    def train_on_optimal_time_windows(self, data, y, n_processes, n_channels=10, filter_channels:bool=True):
        """Train LDA model on the optimal time windows for top performing channels, specified by n_channels"""
        self._set_attributes(data)
        results = self._multiprocessing_time_window_grid_search(data, y, n_processes, filter_channels=filter_channels)

        # Unravel the results from the multiprocessing and sort them by channels
        optimal_time_windows_per_channel = [item for sublist in results for item in sublist]
        optimal_time_windows_per_channel.sort(key=lambda x: x[0])
        optimal_time_windows_per_channel.sort(key=lambda x: x[2], reverse=True)
        self.__optimal_time_windows_per_channel = optimal_time_windows_per_channel

        for channel, times, _ in optimal_time_windows_per_channel[:n_channels]:
            X = self._create_X(data, channel, times)
            self._train(X,y)
        
        self._reshape_attributes((n_channels,-1))


    def train_per_channel_and_timestep(self, data, y, time_resolution):
        """Train an LDA model on each channel and timestep"""
        self._set_attributes(data, time_resolution=time_resolution)

        for channel in range(self.__num_channels):
            for time in range(self.__rescaled_timesteps):
                X = self._create_X(data, channel, time*self.__time_resolution)
                self._train(X, y)

        self._reshape_attributes((self.__num_channels,self.__rescaled_timesteps,-1))

    def train_on_all_channels(self, data, y, time_resolution, filter_channels:bool = False, custom_channels = None):
        """Train an LDA model on all the channels for each timestep"""
        self._set_attributes(data, time_resolution=time_resolution)

        for time in range(self.__rescaled_timesteps):
            if filter_channels:
                self._filter_channels()
                X = self._create_X(data, self._filtered_elec_areas_idxs, time*self.__time_resolution)
            elif not custom_channels == None:
                X = self._create_X(data, custom_channels, time*self.__time_resolution)
            else:
                X = self._create_X(data, np.arange(0,self.__num_channels), time*self.__time_resolution)

            X_reshaped = X.reshape(self.__num_trials,-1)
            self._train(X_reshaped,y)
        
        self._reshape_attributes((self.__rescaled_timesteps,-1))
    
    def train_on_all_timesteps(self, data, y, setup_data):
        """Train an LDA model on all the timesteps for each channel"""
        self._set_attributes(data, setup_data)

        for channel in range(self.__num_channels):
            X = self._create_X(data, channel, np.arange(self.__num_timesteps))
            X_reshaped = X.reshape(self.__num_trials, -1)
            self._train(X_reshaped, y)
        
        self._reshape_attributes((self.__num_channels,-1))

class ShuffledLDA(LDA):
    def __init__(self,
                 setup_data):
        super().__init__()
        self._get_hand(setup_data)

        
    def _get_hand(self, setup_data):
        """Get the subject's card value at a particular trial"""
        bets = setup_data['filters']['bets']
        good_trials = np.where(np.isnan(bets) == False)[0] # extract indices of trials without the 'nan'

        self.__sub_hand = setup_data['filters']['card1'][good_trials] # get the subject's card hand for the good trials
    
    def _shuffle_y(self, y):
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
    
    def train_per_channel_and_timestep(self, data, y, setup_data, time_resolution):
        """Use shuffled y labels to train LDA model on each channel and timestep"""
        y_shuffled = self._shuffle_y(y)
        super().train_per_channel_and_timestep(data, y_shuffled, setup_data=setup_data, time_resolution=time_resolution)

    def compute_t_stat_clusters(self, ref_estimator, threshold):
        """Computes the sum of the t-statistics for a cluster specified by a reference LDA estimator"""
        t_stat_sums = []

        for channel in range(self.num_channels):
            temp_t_stat_sums = []
            for arr in ref_estimator.create_cluster_idxs(threshold)[channel]:
                temp_t_stat_sums.append(self.t_stats[channel][arr].sum())
            
            t_stat_sums.append(temp_t_stat_sums)
        
        return t_stat_sums