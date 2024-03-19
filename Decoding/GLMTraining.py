#%%
import h5py 
import mat73
import numpy as np
import csv

# from LDA import TrainOptimalTimeWindows
from trees_forests import RandomForestOptimal
from helper_functions import plot_freq_band_data_for_channel

# %%
# subs = ['06']
# subs = ['06','07','10','12','13','15','16','17','18','21']
subs = ['06', '13', '16', '17', '18', '21']
# subs = ['07', '10', '12', '15', '16']

file_paths = {}

for sub in subs:
    # Create a dictionary holding the file paths
    ncsl_share = '/mnt/ncsl_share'
    file_paths[sub] = {
        'setup_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat',
        'raw_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat',
        'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy', # movement onset as event,
        # 'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/show-card_pre-2sec_post-4sec/Subject{sub}_snapshot_normalized.npy', # show card as event
        'out_path_metrics': f'Metrics/LeftMiddleTemporalGyrus/PredictCard/RandomForestShowCard/Subject{sub}',
        'out_path_plots': f'Plots/LeftMiddleTemporalGyrus/PredictCard/RandomForestShowCard/Subject{sub}'
    }
## %%

# %%
# Settings for matplotlib graphs
import matplotlib as mpl

mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 18
## %%
#%%
for sub in subs:
    # Load appropriate files/data
    raw_file = h5py.File(file_paths[sub]['raw_path'])
    setup_data = mat73.loadmat(file_paths[sub]['setup_path'])

    out_path_plots = file_paths[sub]['out_path_plots']
    out_path_metrics = file_paths[sub]['out_path_metrics']

    # Instantiate approparite variables  
    bets = setup_data['filters']['bets']
    good_trials = np.where(np.isnan(bets) == False)[0]
    bets = bets[good_trials]
    subject_cards = setup_data['filters']['card1'][good_trials] # get the subject's card values for the good trials

    elec_names = np.array(setup_data['elec_name'])
    elec_areas = np.array(setup_data['elec_area'])

    # # Trying to predict card value from neural data
    # card_categories = []
    # trials_to_keep = []

    # six_card_trials = np.where(subject_cards == 6)[0]

    # for i, card in enumerate(np.delete(subject_cards, six_card_trials)):
    #     if card  < 8:
    #         card_categories.append(0)
    #     else:
    #         card_categories.append(1)

    brain_area = 'middle temporal gyrus L'
    channel_idxs = np.where(elec_areas == brain_area)[0]

    # data = np.load(file_paths[sub]['data_path'])[:,channel_idxs,:,40:80] # 40:80 is the time window of interest for visual cue as event
    # card_categories = np.asarray(card_categories)

    data = np.load(file_paths[sub]['data_path'])[:,channel_idxs,:,:60] # 0:60 is the time window of interest for movement onset as event
    y = np.asarray([(0 if bet == 5 else 1) for bet in bets]) # 0 = low bet ($5), 1 = high bet ($20)

    # Group frequencies into frequency bands
    wavelet_freqs = np.logspace(np.log2(2),np.log2(150),num=63,base=2)

    frequency_band_indices ={
        "Delta" : [i for i,freq in enumerate(wavelet_freqs) if freq >= 0.5 and freq < 4],
        "Theta" : [i for i,freq in enumerate(wavelet_freqs) if freq >= 4 and freq < 8],
        "Alpha" : [i for i,freq in enumerate(wavelet_freqs) if freq >= 8 and freq < 14],
        "Beta" : [i for i,freq in enumerate(wavelet_freqs) if freq >= 14 and freq < 30],
        "Gamma" : [i for i,freq in enumerate(wavelet_freqs) if freq >= 30]
    }

    f_band_data = np.zeros((data.shape[0], data.shape[1], 5, data.shape[3]))

    for i, key in enumerate(frequency_band_indices):
        f_band_data[:,:,i,:] = data[:,:,frequency_band_indices[key],:].mean(2)
    
    clf = RandomForestOptimal(data=f_band_data, setup_data=setup_data)
    clf._elec_names = elec_names[channel_idxs]
    clf._elec_areas = elec_areas[channel_idxs]

    if data.shape[1] < 20:
        num_processes = data.shape[1]
    else:
        num_processes = 20
    
    if data.shape[1] < 10:
        num_channels = data.shape[1]
    else:
        num_channels = 10

    # filtered_num_channels = clf.filter_channels()[2]
    clf.train_on_optimal_time_windows(f_band_data, y, n_processes=num_processes, n_channels=num_channels)

    ch_names = []
    optimal_time_windows_start = []
    optimal_time_windows_end = []
    accuracies = []

    for optimal_time_window_info in clf._optimal_time_windows_per_channel:
        ch_idx = optimal_time_window_info[0]
        ch_names.append(clf._elec_names[ch_idx])
        optimal_time_windows_start.append(optimal_time_window_info[1][0])
        optimal_time_windows_end.append(optimal_time_window_info[1][1])
        accuracies.append(optimal_time_window_info[2])

    info = zip(ch_names, optimal_time_windows_start, optimal_time_windows_end, accuracies)

    brain_area = brain_area.replace(' ', '_')
    file_name = f'Metrics/RightMiddleTemporalGyrus/PredictBet/RandomForestMovementOnset/ChannelInfo/Subject{sub}_{brain_area}_info.csv'

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Channel Names', 'Optimal Time Windows Start', 'Optimal Time Windows End', 'Accuracy'])  # Write header
        writer.writerows(info)  # Write data rows

    # for optimal_time_windows_info in clf._optimal_time_windows_per_channel[:10]:
    #     ch = optimal_time_windows_info[0]
    #     time_window = optimal_time_windows_info[1]

    #     # Plot the data for the optimal time window for each channel
    #     plot_freq_band_data_for_channel(f_band_data, y, ch, time_window, event_delay=3, out_path_plots=out_path_plots, elec_names=elec_names, elec_areas=elec_areas)

    # accuracies, peak_accuracy_group_idx = clf.get_group_accuracies(y)
    # top_chs = [ch for ch, _, _ in clf._optimal_time_windows_per_channel[:peak_accuracy_group_idx+1]]
    # clf.plot_accuracies(y, out_path_plots)
    # clf.plot_heatmap(top_chs, 0, accuracies[peak_accuracy_group_idx], False, out_path_plots)
    # optimal_time_window_channel_combination = elec_areas[top_chs]
    # np.save(out_path_metrics + '_top_n_channel_combination.npy', optimal_time_window_channel_combination)

    # peak_accuracy, optimal_channel_combination = clf.get_optimal_channel_combination(y, n_channels=num_channels)
    # # clf.plot_freq_box_plots(y, results[0][0], out_path_plots)
    # clf.plot_heatmap(optimal_channel_combination, 0, sub, peak_accuracy, True, out_path_plots)
    # np.save(out_path_metrics + '_optimal_channel_combination.npy', elec_areas[optimal_channel_combination])

# %%
# %%
