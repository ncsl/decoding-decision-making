# %%
import h5py
import mat73
import numpy as np

from SinkSourceConnectivity.sink_source import SinkSource

SI_t_test = []

# %%
subs = ['06','07','10','12','13','15','16','17','18','21']
# subs = ['07']
file_paths = {}

for sub in subs:
    # create a dictionary holding the file paths
    ncsl_share = '/mnt/ncsl_share'
    file_paths[sub] = {
        'setup_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat',
        'raw_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat',
        'out_path_plots': f'Heatmaps/Subject{sub}',
    }

    raw_file = h5py.File(file_paths[sub]['raw_path'])
    setup_data = mat73.loadmat(file_paths[sub]['setup_path'])

    ss = SinkSource(raw_file, setup_data)

    data = ss.get_data(raw_file)
    y = ss.get_y()

    high_bet_trials = np.where(y == 1)[0]
    low_bet_trials = np.where(y == 0)[0]

    A_hat_all = ss.estimateA_all_subjects(data)

    fs = 500
    winSize_sec = 0.2

    sink_row = 1
    sink_col = 1
    winSize = winSize_sec*fs

    onset_delay = 3
    dsFs = 500
    window_length = data.shape[2]/dsFs
    num_timesteps = A_hat_all.shape[3]
    time = np.linspace(0,window_length,num_timesteps) - onset_delay
    time = [round(i, 2) for i in time]

    A_win_all, SI_wins_all, _, _ = ss.computeSS(A_hat_all)

    SI_wins_high_bet = SI_wins_all[high_bet_trials,:,:]
    SI_wins_low_bet = SI_wins_all[low_bet_trials,:,:]

    SI_wins_difference = SI_wins_high_bet.mean(0) - SI_wins_low_bet.mean(0)

    SI_wins_sorted, labels_sort = ss.sort_SI_wins(A_hat_all, SI_wins_difference)

    # t_stats, p_vals, unique_ch_areas = ss.SI_t_test(SI_wins_all, y)
    # SI_t_test.append([unique_ch_areas, t_stats, p_vals])

    # ss.plot_p_vals(SI_wins_all, y, out_path_plots=file_paths[sub]['out_path_plots'])
    # ss.plot_t_stats(SI_wins_all, y, out_path_plots=file_paths[sub]['out_path_plots'])
    ss.plot_SI_heatmap(A_hat_all, SI_wins_difference, winSize=winSize, fs=fs, time=time, out_path_plots=file_paths[sub]['out_path_plots'])

# %%
np.save('SI_t_test.npy', SI_t_test)
# %%
