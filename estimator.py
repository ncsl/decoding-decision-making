import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from multiprocessing import Pool
from helper_functions import _generate_sampled_channels, _find_combinations, _get_collective_predictions, _get_collective_prediction_accuracy

from abc import ABC, abstractmethod

class Estimator(ABC):
    def __init__(self, data, setup_data):
        self._num_trials, self._num_channels, self._num_freqs, self._num_timesteps = data.shape
        
        self._elec_areas = setup_data['elec_area']
        self._elec_names = setup_data['elec_name']

        self._reset_metrics()

    def _reset_metrics(self):
        self.mean_scores = []
        self.dvals = []
        self.low_bet_powers = []
        self.high_bet_powers = []
        self.diff_avg_powers = []

    def _reshape_attributes(self, new_shape:tuple):
        """Reshape class attributes to specified shape"""
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, np.reshape(getattr(self, attr_name), new_shape))

    def create_X(self, data, channel, time):
        """Create the X data that will be used to train the LDA model"""
        if hasattr(self, '_time_resolution') and type(self._time_resolution) == int:
            X = data[:, channel, :, time:time+self._time_resolution].mean(-1)
        elif type(time) == list and len(time) == 2:
            if time[0] - time[1] == 0:
                X = data[:, channel, :, time[0]]
            else:
                X = data[:, channel, :, time[0]:time[1]].mean(-1)
        else:
            pass

        return X
    
    def filter_channels(self):
        """Filters out channels that are in particular anatomical locations"""
        filtered_elec_areas_idxs = [i for i,ea in enumerate(self._elec_areas) if ea not in 
                                    ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid',
                                     'lesion L','ventricle L','ventricle R']]
        filtered_elec_areas = [self._elec_areas[i] for i in filtered_elec_areas_idxs]
        filtered_num_channels = len(filtered_elec_areas_idxs)

        return filtered_elec_areas_idxs, filtered_elec_areas, filtered_num_channels
    
    def set_attributes(self, **kwargs):
        """Set class attributes specified by the dataset and metadata"""
        if 'time_resolution' in kwargs:
            if not(self._num_timesteps % kwargs['time_resolution'] == 0):
                raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
            else:
                self._time_resolution = kwargs['time_resolution']
                self._timesteps_rescaled = int(self._num_timesteps/kwargs['time_resolution'])

class EstimatorTrainOptimalTimeWindows(Estimator):
    @abstractmethod
    def train(self):
        pass

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
        """Train logistic regression model on all possible time windows, 
        store the time windows that correspond with the highest accuracy."""

        time_windows = self._create_time_windows()
        best_time_windows = []

        for channel in channels:
            for times in time_windows:
                X = super().create_X(data, channel, times)
                self.train(X,y)

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

    def get_optimal_channel_combination(self, y, max_channels=10):
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

        if out_path is not None:
            plt.savefig(out_path + '_optimal_time_window_all_group_accuracies.png')
            plt.show()
    
    def plot_freq_box_plots(self, y:np.ndarray, channels:list, out_path:str=None):
        """Plot box-and-whisker plots of the power of each frequency band for each channel in the optimal combination"""
        optimal_time_window_info_channels = [lst[0] for lst in self._optimal_time_windows_per_channel]
        for ch in channels:
            idx = np.where(optimal_time_window_info_channels == ch)[0][0]

            bet_powers = np.concatenate((self.low_bet_powers[idx], self.high_bet_powers[idx])).flatten()

            powers = {
                'Z-Scored Powers' : bet_powers,
                'Frequency Band' : np.tile(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], int(len(bet_powers) / 5)),
                'Category' : np.concatenate((np.repeat('Low Bet', 5*(len(y) - y.sum())), np.repeat('High Bet', 5*y.sum())))
            }

            power_df = pd.DataFrame(powers)

            fig, axs = plt.subplots(1, 1, figsize=(20, 5))
            sns.boxplot(data=power_df, x='Frequency Band', y='Z-Scored Powers', hue='Category', ax=axs)
            axs.set_title(f'Channel {ch} - {self._elec_areas[ch]}')

            if out_path is not None:
                plt.savefig(out_path + f'_power_per_freq_box_plots_{ch}_{self._elec_areas[ch]}.png')
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
        
        axs.set_ylabel('Channel')
        axs.set_xlabel('Time (s)')
        axs.set_xticks(np.arange(0, self._num_timesteps, 5))
        axs.set_xticklabels(time[::5])
        axs.set_yticks(np.arange(len(channels))+0.5)
        axs.set_yticklabels(np.asarray(self._elec_areas)[channels_sorted_by_time_window], rotation = 0)
        axs.axvline(np.argmin(np.abs(time)) + 1, color = 'blue', alpha=1, ls = '--')

        axs.tick_params(axis='y', pad=25)
        if out_path is not None:
            if optimal_combination:
                axs.set_title(f'Accuracy of Optimal Combination of Channels for a Given Time Window\nAccuracy: {top_accuracy:.2f}')
                path = out_path + '_optimal_time_window_and_combination_heatmap.png'
            else:
                axs.set_title(f'Accuracy of Top {len(channels)} Channels for a Given Time Window\nAccuracy: {top_accuracy:.2f}')
                path = out_path + '_optimal_time_window_heatmap.png'
            plt.savefig(path, bbox_inches='tight')
            plt.show()

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
            self.train(X,y)
        
        super()._reshape_attributes((n_channels,-1))

