import numpy as np
import itertools
from matplotlib import pyplot as plt
import seaborn as sns

def _convert_timesteps_to_time(event_delay, time_resolution, timesteps:np.ndarray):
    """Convert timesteps into seconds while specifying timepoint 0."""
    # time point "0 seconds" denoted by event_delay
    times = timesteps/(20/time_resolution) - event_delay 
    times = [round(time, 1) for time in times]
    return times


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

def _find_combinations(n, k):
    """Find all possible combinations of k channels from n channels

    Return
    ------
    combinations_list : list
        List of tuples containing the indices all possible combinations of k channels from n channels
    """
    population = list(range(0, n))
    combinations_list = list(itertools.combinations(population, k))
    return combinations_list

def _get_collective_predictions(predictions):
    # Get the collective prediction for each trial
    collective_predictions = []

    for trial_predictions in predictions:
        if trial_predictions.mean() >= 0.5:
            collective_predictions.append(1)
        else:
            collective_predictions.append(0)
    
    return collective_predictions

def _get_collective_prediction_accuracy(collective_predictions, y):
    # Get the accuracy of theveloped a circuit model of decision-making which accounts for the specificity of inputs to and outputs from inhibitory neurons. We found that selective inhibition expands the space of circuits supporting decision-making, allowing for weaker or stronger recurrent excitation when connected in a competitive or feedback motif. The specificity of inhibitory outputs sets te collective prediction
    accuracy = (y == collective_predictions).mean()
    return accuracy

def plot_freq_band_data_for_channel(f_band_data, y, channel, time_window=None, event_delay=0, time_resolution=1, f_band_labels=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], out_path_plots=None, elec_areas=None):
    low_bet_f_band_powers = f_band_data[np.where(y==0), channel][0].mean(0)
    high_bet_f_band_powers = f_band_data[np.where(y==1), channel][0].mean(0)
    diff_f_band_powers = high_bet_f_band_powers-low_bet_f_band_powers

    xticks = np.arange(0,diff_f_band_powers.shape[1],2)

    fig, axs = plt.subplots(1,1, figsize=(15,10))
    sns.heatmap(diff_f_band_powers, cmap='PRGn', vmin=-0.5, vmax=0.5, ax=axs, cbar_kws={"label":"Z-Scored Power"})
    axs.set_yticklabels(f_band_labels)
    axs.set_xticks(xticks)
    axs.set_xticklabels(labels=_convert_timesteps_to_time(event_delay, time_resolution, xticks), rotation = 90)
    plt.gca().invert_yaxis()

    axs.set_title(f'Difference in Average Frequency Band Powers (High Bet - Low Bet)\nChannel {channel} - {elec_areas[channel]}')

    axs.set_ylabel('Frequency Band')
    axs.set_xlabel('Time (s)')

    for label in axs.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    if time_window is not None:
        if time_window[0]==0:
            plt.axvline(x=1, color='red')
            plt.axvline(x=time_window[1], color='red')
        else:
            plt.axvline(x=time_window[0], color='red')
            plt.axvline(x=time_window[1], color='red')
    
    if out_path_plots is not None:
        plt.savefig(out_path_plots + '_channel_' + str(channel) + '_freq_band_data.png')