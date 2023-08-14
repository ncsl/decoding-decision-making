from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import RepeatedKFold
from estimator import Estimator
from helper_functions import _generate_sampled_channels, _find_combinations, _get_collective_predictions, _get_collective_prediction_accuracy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

class LogisticReg(Estimator):
    """Class to train Logistic Regression model on data"""
    def __init__(self, data, setup_data):
        super().__init__(data, setup_data)
        self._reset_metrics()

    def _reset_metrics(self):
        self.mean_scores = []
        self.dvals = []
        self.low_bet_avg_powers = []
        self.high_bet_avg_powers = []
        self.diff_avg_powers = []

    def _reshape_attributes(self, new_shape:tuple):
        """Reshape class attributes to specified shape"""
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, np.reshape(getattr(self, attr_name), new_shape))

    def train(self, X, y):
        low_bet_avg_powers = X[np.where(y == 0)].mean(0)
        high_bet_avg_powers = X[np.where(y == 1)].mean(0)
        diff_avg_powers = high_bet_avg_powers - low_bet_avg_powers

        self.high_bet_avg_powers.append(high_bet_avg_powers)
        self.low_bet_avg_powers.append(low_bet_avg_powers)
        self.diff_avg_powers.append(diff_avg_powers)


        # Using RepeatedKFold() for training Logistic Regression model
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

        estimators = []
        scores = []
        dval = np.zeros(self._num_trials)

        for train, test in rkf.split(X):
            clf = LogisticRegression(random_state=0).fit(X[train], y[train])
            estimators.append(clf)
            scores.append(clf.score(X[test], y[test]))
            dval[test] = np.dot(X[test], clf.coef_.T).T[0] + clf.intercept_
            
        self.mean_scores.append(np.mean(scores))
        self.dvals.append(dval)

class TrainOptimalTimeWindows(LogisticReg):
    def __init__(self, data, setup_data) -> None:
        super().__init__(data, setup_data)

    def _create_time_windows(self):
        """Create all possible time windows from which X data can be created."""
        time_windows = []
        for i in range(self._num_timesteps):
            for j in range(self._num_timesteps):
                if i-j >= 0 and i+j <= self._num_timesteps:
                    time_windows.append([i-j,i+j])
                else:
                    break
        
        return time_windows
    
    def _get_predictions(self):
        predictions = []

        # Get predictions for each trial by each channel
        for trial in range(self._num_trials):
            trial_predictions = []
            for dval in self.dvals[:,trial]:
                if dval >= 0:
                    channel_prediction = 1
                else:
                    channel_prediction = 0
                trial_predictions.append(channel_prediction)
            
            predictions.append(trial_predictions)

        predictions = np.asarray(predictions)

        return predictions
    
    def _grid_search_on_channel_combinations(self, y, max_channels=20):
        assert max_channels <= 20, 'Cannot perform grid search on more than 20 channels'

        predictions = self._get_predictions()
            # Find all possible channel combinations to use for collective prediction
        all_channel_idxs_combinations = []
        for i in range(max_channels):
            # Find all possible channel combinations of length i+1
            channel_combinations = _find_combinations(max_channels, i+1)
            all_channel_idxs_combinations.append(channel_combinations)

        # Grid search on optimal channel combination to use for collective prediction
        accuracies = []

        for k_length_combinations in all_channel_idxs_combinations:
            k_length_combination_accuracies = []
            for combination in k_length_combinations:
                # Stores all the predictions for this particular combination of k channels
                combination = list(combination)
                combination_predictions = predictions[:,combination]

                # Stores the collective prediction for this particular combination of k channels
                collective_combination_predictions = _get_collective_predictions(combination_predictions)

                # Get the accuracy of the collective prediction for each combination of k channels
                k_length_combination_accuracies.append(_get_collective_prediction_accuracy(collective_combination_predictions, y))
            
            accuracies.append(k_length_combination_accuracies)

        return all_channel_idxs_combinations, accuracies

    def _time_window_grid_search(self, data, y, channels):
        """Train LDA model on all possible time windows, store the time windows that correspond with the highest LDA score."""

        time_windows = self._create_time_windows()
        best_time_windows = []

        for channel in channels:
            for times in time_windows:
                X = super().create_X(data, channel, times)
                super().train(X,y)

            print(f'Channel {channel} done')
            best_time_windows.append([channel, time_windows[np.argmax(self.mean_scores)], np.max(self.mean_scores)])
            super()._reset_metrics()
        
        return best_time_windows

    def _multiprocessing_time_window_grid_search(self, data, y, n_processes, filter_channels:bool = True):
        """Perform a time window grid search in parallel"""
        if filter_channels:
            filtered_elec_areas_idxs, _, __ = super().filter_channels()
            channels = filtered_elec_areas_idxs
        else:
            channels = np.arange(self._num_channels)

        sample_size = round(len(channels)/n_processes)

        sampled_channels = _generate_sampled_channels(channels, sample_size, [])

        # if __name__ == '__main__':
        with Pool(n_processes) as p:
            results = p.starmap(self._time_window_grid_search, [(data, y, channels) for channels in sampled_channels])
            p.close()
        
        return results

    def train_on_optimal_time_windows(self, data, y, n_processes, n_channels=10, filter_channels:bool=True):
        """Train LDA model on the optimal time windows for top performing channels, specified by n_channels"""
        results = self._multiprocessing_time_window_grid_search(data, y, n_processes, filter_channels=filter_channels)

        # Unravel the results from the multiprocessing and sort them by channels
        optimal_time_windows_per_channel = [item for sublist in results for item in sublist]
        optimal_time_windows_per_channel.sort(key=lambda x: x[0])
        optimal_time_windows_per_channel.sort(key=lambda x: x[2], reverse=True)
        self._optimal_time_windows_per_channel = optimal_time_windows_per_channel

        for channel, times, _ in optimal_time_windows_per_channel[:n_channels]:
            X = super().create_X(data, channel, times)
            super().train(X,y)
        
        super()._reshape_attributes((n_channels,-1))

    def get_group_accuracies(self, y):
        """Get all the collective prediction accuracies of top channels for all group sizes"""
        predictions = self._get_predictions()
        accuracies = []

        for i in range(predictions.shape[1]):
            # Get collective prediction of channels 0 to i
            collective_predictions = _get_collective_predictions(predictions[:,:i+1])
            accuracies.append(_get_collective_prediction_accuracy(collective_predictions, y))
        
        peak_accuracy_group_idx = np.argmax(accuracies)

        return accuracies, peak_accuracy_group_idx

    def get_optimal_channel_combination(self, y, max_channels=20):
        """
        Get the optimal channel combination to use for collective prediction. 
        Channels used to find combination is specified by max_channels.
        """

        all_channel_idxs_combinations, accuracies = self._grid_search_on_channel_combinations(y, max_channels=max_channels)

        optimal_time_windows_per_ch = self._optimal_time_windows_per_channel
        max_accuracies = []

        for i, accuracies in enumerate(accuracies):
            optimal_chs = []
            ch_idxs = all_channel_idxs_combinations[i][np.argmax(accuracies)]
            for idx in ch_idxs:
                optimal_chs.append(optimal_time_windows_per_ch[idx][0])

            max_accuracies.append([optimal_chs, np.max(accuracies)])

        max_accuracies.sort(key=lambda x: x[1], reverse=True)

        return max_accuracies
    
    def plot_accuracies(self, y, out_path:str=None):
        """Plot the collective prediction accuracy for each group size of top performing channels"""
        accuracies, peak_accuracy_group_idx = self.get_group_accuracies(y)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(np.arange(len(accuracies)) + 1, accuracies)
        ax.set_title('Accuracy of Majority Concensus')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Number of Top Performing Channels in Group')
        ax.axvline(peak_accuracy_group_idx + 1, color = 'red', alpha=0.5)
        ax.annotate(f'(Group Size for Peak Accuracy: {peak_accuracy_group_idx + 1}\nScore: {accuracies[peak_accuracy_group_idx]:.2f})', xy=(peak_accuracy_group_idx,np.mean(accuracies)), fontsize = 12)

        plt.savefig(out_path + '_optimal_time_window_all_group_accuracies.png')
        plt.show()

    def plot_heatmap(self, channels:list, event_delay:int, top_accuracy:float = None, optimal_combination:bool=False, out_path:str=None):
        """
        Plot a heatmap of the accuracy of the selected channels for their respective time windows.
        Heatmap is sorted by the accuracy of the time window.
        """

        # Convert the number of time steps to seconds
        time = np.arange(self._num_timesteps)/20 - event_delay # 20 is the number of timesteps per second

        optimal_time_window_info_for_channels = []

        for channel in channels:
            optimal_time_window_info = [lst for lst in self._optimal_time_windows_per_channel if lst[0] == channel]
            optimal_time_window_info_for_channels.append(optimal_time_window_info[0])

        # Ensures that channels are sorted by the start time of their time windows
        optimal_time_window_info_for_channels.sort(key=lambda x: x[1][1])
        optimal_time_window_info_for_channels.sort(key=lambda x: x[1][0])

        channels_sorted_by_time_window = [lst[0] for lst in optimal_time_window_info_for_channels]
        print(channels_sorted_by_time_window)

        accuracies = []

        heatmap_array = np.zeros((len(channels), self._num_timesteps))

        # Get the locations of the channels from the data strucure
        
        for i, (channel, time_window, accuracy) in enumerate(optimal_time_window_info_for_channels):
            # If time_window is exactly at one timestep, visualize the time window to be one timestep larger
            if time_window[0] - time_window[1] == 0:
                time_window = [time_window[0], time_window[1]+1]

            heatmap_array[i,time_window[0]:time_window[1]] = accuracy
            accuracies.append(accuracy)

        fig, axs = plt.subplots(1, 1, figsize=(10, 25))
        sns.heatmap(heatmap_array, ax=axs, cmap='PRGn', vmin=np.min(accuracies)-.05, vmax=np.max(accuracies), center=np.min(accuracies)-.05, cbar_kws={"label":"Channel Accuracy"})
        
        if optimal_combination:
            axs.set_title(f'Accuracy of Optimal Combination of Channels for a Given Time Window\nAccuracy: {top_accuracy:.2f}')
            path = out_path + '_optimal_time_window_and_combination_heatmap.png'
        else:
            axs.set_title(f'Accuracy of Top {len(channels)} Channels for a Given Time Window\nAccuracy: {top_accuracy:.2f}')
            path = out_path + '_optimal_time_window_heatmap.png'
        
        axs.set_ylabel('Channel')
        axs.set_xlabel('Time (s)')
        axs.set_xticks(np.arange(0, self._num_timesteps, 5))
        axs.set_xticklabels(time[::5])
        axs.set_yticks(np.arange(len(channels))+0.5)
        axs.set_yticklabels(np.asarray(self._elec_areas)[channels_sorted_by_time_window], rotation = 0)
        axs.axvline(np.where(time == 0), color = 'blue', alpha=1, ls = '--')

        axs.tick_params(axis='y', pad=25)

        plt.savefig(path, bbox_inches='tight')
        plt.show()