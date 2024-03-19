import numpy as np
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import mne
import re
from scipy.signal import convolve2d
from scipy.stats import zscore

def _convert_timesteps_to_time(event_delay, time_resolution, timesteps:np.ndarray):
    """Convert timesteps into seconds while specifying timepoint 0."""
    # time point "0 seconds" denoted by event_delay
    times = timesteps/(20/time_resolution) - event_delay 
    times = [round(time, 1) for time in times]
    return times


def _generate_multiprocessing_groups(channels, group_size, grouped_channels=[]):
    """Recursively organize channels into groups to be used for multiprocessing"""
    np.random.seed()

    if len(channels) > group_size:
        sample = np.random.choice(channels, size=group_size, replace=False)
        grouped_channels.append(list(sample))
        channels = np.delete(channels, np.where(np.isin(channels, sample))[0])
        _generate_multiprocessing_groups(channels, group_size, grouped_channels)
    else:
        grouped_channels.append(list(channels))

    return grouped_channels

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
        if trial_predictions.mean() > 0.5:
            collective_predictions.append(1)
        else:
            collective_predictions.append(0)
    
    return collective_predictions

def _get_collective_prediction_accuracy(collective_predictions, y):
    # Get the accuracy of theveloped a circuit model of decision-making which accounts for the specificity of inputs to and outputs from inhibitory neurons. We found that selective inhibition expands the space of circuits supporting decision-making, allowing for weaker or stronger recurrent excitation when connected in a competitive or feedback motif. The specificity of inhibitory outputs sets te collective prediction
    accuracy = (y == collective_predictions).mean()
    return accuracy

def laplacian_reference(elec_names, Fs, raw_file):
    lfp_data = raw_file['lfpdata']
    lfp_all = lfp_data[:,:]

    # Filter Data Bandpass .5-200 Hz 
    filt = mne.filter.filter_data(lfp_all,Fs,0.5,200,method="iir")
    # notch filter 60 hz harmonics
    for notchfreq in [60,120,180]:
        filt = mne.filter.notch_filter(filt,Fs,notchfreq, method="iir")
    # decimate to 500 Hz 
    decFactor = int(Fs/500)
    filt = filt[:,::decFactor]

    ## For each channel in elec_names, get its index position in array, whether its on the end of the electrode shaft, and its neighboring indices 
    lap_ref_data = np.zeros(filt.shape)
    for i,en in enumerate(elec_names):
        if en in ["REF1","REF2","E","CZ","FZ","PZ"]:
            lap_ref_data[i,:] = filt[i,:]
            continue
        pattern = r"([a-z']+) *(\d+)"
        shaft_name = re.findall(pattern,en,re.IGNORECASE)[0][0]
        elec_num = re.findall(pattern,en,re.IGNORECASE)[0][1]
        en_plus1 = f"{shaft_name}{str(int(elec_num)+1)}"
        en_minus1 = f"{shaft_name}{str(int(elec_num)-1)}"
        if en_minus1 not in elec_names:
            neighbor_inds=[i-1]
        elif en_plus1 not in elec_names:
            neighbor_inds=[i+1]
        else:
            neighbor_inds = [ i-1,i+1]
        print(en, i, neighbor_inds,[elec_names[n] for n in neighbor_inds])
        neighbor_mean = np.mean(filt[neighbor_inds,:],axis=0)
        lap_ref_data[i,:] = filt[i,:] - neighbor_mean
    
    return lap_ref_data

def _wavelet_transform(elec_names, Fs, raw_file):
    lap_ref_data = laplacian_reference(elec_names, Fs, raw_file)
    wavlet_freqs = np.logspace(np.log2(2),np.log2(150),num=63,base=2)
    #tfr_by_chan_list = []    
    tfr_array = np.zeros((lap_ref_data.shape[0], len(wavlet_freqs), int(lap_ref_data.shape[1]/25)))
    for ch in range(lap_ref_data.shape[0]):
        tfr = mne.time_frequency.tfr_array_morlet(lap_ref_data[ch,:].reshape(1,1,-1),500,wavlet_freqs,n_cycles=6,output='power',n_jobs=1,)
        #
        #downsample = np.convolve(tfr[0,0,0,:], np.ones(50, ), mode='valid')[::25]/50. #Fs = 500; 1/Fs = .002 s; .1 sec/.002 sec = 50 samples for 100 ms. hence 50. then downsample every 25 because going from 500 Hz to 20 Hz final time resolution
        downsample = convolve2d(tfr[0,0,:,:], (1.0/50)* np.ones((1,50)), mode='same')[:,::25]
        #tfr_by_chan_list.append(downsample)
        downsample = np.log(downsample) # natural log normalization to make frequencies more comparable
        downsample = zscore(downsample,axis=1)
        tfr_array[ch,:,:] = downsample

    return tfr_array

def snapshot_data(setup_data, raw_file, elec_names, Fs, event_id:int, time_interval:list):
    tfr_array = _wavelet_transform(elec_names, Fs, raw_file)

    window_length = np.abs(time_interval[0])+ np.abs(time_interval[1])

    ## Snapshot around start move 
    dsFs = 20 #downsample FS is 20 Hz 
    good_trials = setup_data['filters']['trial'][setup_data['filters']['success']].astype(int)-1
    num_trials = len(good_trials)
    trials_by_channels_by_freqs_by_time_array = np.zeros((num_trials,tfr_array.shape[0],tfr_array.shape[1],int(window_length*dsFs)))
    for i,t in enumerate(good_trials):
        event_time = setup_data['trial_times'][t][0][setup_data['trial_words'][t][0]==event_id][0]
        #print(f'start move time = {start_move_time} for trial {t}')
        
        ## To go from the time to the index position in the lfp array, multiply time by Fs 
        # start_move_index = int(event_time*dsFs)
        #print(f'start move index = {start_move_index} for trial {t}')
        start_index = int((event_time + time_interval[0])*dsFs)
        end_index = start_index+int(window_length*dsFs)
        data_slice = tfr_array[:,:,start_index:end_index]
        trials_by_channels_by_freqs_by_time_array[i,:,:,:] = data_slice

    return trials_by_channels_by_freqs_by_time_array


def plot_freq_band_data_for_channel(f_band_data, y, ch, time_window=None, event_delay=0, time_resolution=1, f_band_labels=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], out_path_plots=None, elec_names=None, elec_areas=None):
    low_bet_f_band_powers = f_band_data[np.where(y==0), ch][0].mean(0)
    high_bet_f_band_powers = f_band_data[np.where(y==1), ch][0].mean(0)
    diff_f_band_powers = high_bet_f_band_powers-low_bet_f_band_powers

    xticks = np.arange(0,diff_f_band_powers.shape[1],2)

    fig, axs = plt.subplots(1,1, figsize=(15,10))
    sns.heatmap(diff_f_band_powers, cmap='PRGn', vmin=-0.5, vmax=0.5, ax=axs, cbar_kws={"label":"Z-Scored Power"})
    axs.set_yticklabels(f_band_labels)
    axs.set_xticks(xticks)
    axs.set_xticklabels(labels=_convert_timesteps_to_time(event_delay, time_resolution, xticks), rotation = 90)
    plt.gca().invert_yaxis()

    axs.set_title(f'Difference in Average Frequency Band Powers (High Bet - Low Bet)\nChannel {elec_names[ch]} - {elec_areas[ch]}')

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
        plt.savefig(out_path_plots + f'_freq_power_{elec_names[ch]}_{elec_areas[ch]}.png')