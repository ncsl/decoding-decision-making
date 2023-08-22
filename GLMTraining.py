#%%
import h5py 
import mat73
import numpy as np

# from LDA import TrainOptimalTimeWindows
from LDA import TrainOptimalTimeWindows

# %%
# subs = ['21']
subs = ['06','07','10','12','13','15','16','17','18','21']
# subs = ['06', '07']
file_paths = {}

for sub in subs:
    # Create a dictionary holding the file paths
    ncsl_share = '/mnt/ncsl_share'
    file_paths[sub] = {
        'setup_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat',
        'raw_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat',
        'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy', # movement onset as event,
        # 'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/show-card_pre-2sec_post-4sec/Subject{sub}_snapshot_normalized.npy', # show card as event
        'out_path_metrics': f'Metrics/Subject{sub}_lda_movement',
        'out_path_plots': f'Plots/Subject{sub}_lda_movement'
    }
## %%

# %%
# Settings for matplotlib graphs
import matplotlib as mpl

mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
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

    data = np.load(file_paths[sub]['data_path'])[:,:,:,:60] # 0:60 is the time window of interest for movement onset as event
    # data = np.load(file_paths[sub]['data_path'])[:,:,:,40:80] # 40:80 is the time window of interest for show card as event
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
    
    clf = TrainOptimalTimeWindows(data=f_band_data, setup_data=setup_data)
    filtered_num_channels = clf.filter_channels()[2]
    clf.train_on_optimal_time_windows(f_band_data, y, n_processes=20, n_channels=filtered_num_channels)

    accuracies, peak_accuracy_group_idx = clf.get_group_accuracies(y)
    top_chs = [ch for ch, _, _ in clf._optimal_time_windows_per_channel[:peak_accuracy_group_idx+1]]
    clf.plot_accuracies(y, out_path_plots)
    clf.plot_heatmap(top_chs, 3, accuracies[peak_accuracy_group_idx], False, out_path_plots)
    optimal_time_window_channel_combination = elec_areas[top_chs]
    np.save(out_path_metrics + '_top_n_channel_combination.npy', optimal_time_window_channel_combination)

    results = clf.get_optimal_channel_combination(y, max_channels=10)
    clf.plot_freq_box_plots(y, results[0][0], out_path_plots)
    clf.plot_heatmap(results[0][0], 3, results[0][1], True, out_path_plots)
    optimal_time_window_optimal_channel_combination = elec_areas[results[0][0]]
    np.save(out_path_metrics + '_optimal_channel_combination_10.npy', optimal_time_window_optimal_channel_combination)
## %%
# %%
