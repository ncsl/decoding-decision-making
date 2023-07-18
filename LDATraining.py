#%%
import h5py 
import mat73
import numpy as np

from LDA import LDA
from PlotLDAMetrics import PerChannelTimestep, PerTimestepAllChannels

# %%
# subs = ['06','07','10','12','13','15','16','17','18','21']
subs = ['06', '07']
file_paths = {}

for sub in subs:
    # Create a dictionary holding the file paths
    ncsl_share = '/mnt/ncsl_share'
    file_paths[sub] = {
        'setup_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat',
        'raw_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat',
        'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy',
        'out_path_metrics': f'Metrics/Subject{sub}',
        'out_path_plots': f'Plots/Subject{sub}'
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

    data = np.load(file_paths[sub]['data_path'])
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
    
    lda = LDA()
    lda.train_per_channel_and_timestep(f_band_data, y, 5)
    plots = PerChannelTimestep(lda, setup_data, True)
    plots.plot_sorted_scores(out_path_plots)
    plots.plot_sorted_scores_per_channel(5, out_path_plots)
    plots.plot_power_heatmap(lda.mean_scores,5,out_path_plots)
    
## %%
# %%
