#%%
import os
import h5py 
import mat73
import numpy as np
from matplotlib import pyplot as plt
import re
import mne
from scipy import signal,stats
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%
#setup_path = '/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT06/EFRI06_WAR_SES1_Setup.mat'
#raw_path = '/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT06/EFRI06_WAR_SES1_Raw.mat'

subs = ['06','07','10','12','13','15','16','17','18','21']
subs = ['10']
for sub in subs:
    ncsl_share = '/run/user/1006/gvfs/smb-share:server=10.162.37.21,share=main'
    setup_path = ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat'
    raw_path = ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat'
    data_path = f'Data/Subject{sub}_snapshot_normalized.npy'
    out_path = 'Top_Ten_Accuracy_Graphs'
    # data_path = f'/run/user/1006/gvfs/smb-share:server=10.162.37.21,share=main/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy'
    # out_path = '/run/user/1006/gvfs/smb-share:server=10.162.37.21,share=main/\'Daniel Wang\'/Top_Ten_Accuracy_Graphs'

    raw_file = h5py.File(raw_path)
    setup_data = mat73.loadmat(setup_path)
    data = np.load(data_path, allow_pickle=True)

    num_channels = data.shape[1]
    num_timesteps = data.shape[3]

    bets = setup_data['filters']['bets']
    y = np.asarray([(0 if bet == 5 else 1) for bet in bets if not np.isnan(bet)])

    model_accuracies = np.zeros((num_channels, num_timesteps, 5))  # Number of cross-validation folds (e.g., 5)

    for channel in range(num_channels):
        for time in range(num_timesteps):
            X = data[:, channel, :, time]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            cv_results = cross_validate(lda, X, y, cv=5)
            model_accuracies[channel, time] = cv_results['test_score']

    mean_accuracies = np.mean(model_accuracies, axis=(2)) # Create an array storing the average accuracies for each channel at each timepoint
    mean_accuracies_max = np.zeros((num_channels,2))

    for channel in range(num_channels):
        mean_accuracies_max[channel, 0] = mean_accuracies[channel,:].max()
        mean_accuracies_max[channel, 1] = list(mean_accuracies[channel]).index(mean_accuracies[channel,:].max()) # get the time point at which the maximum accuracy occurs

    top_ten_accuracies_index = np.argsort(mean_accuracies_max[:,0])[-10:] # store the channel indices of the top 10 accuracies

    # Plot the mean accuracy vs time graph of the top 10 channels with the highest accuracy
    fig, axs = plt.subplots(10, 1, figsize=(24, 48))

    time = np.arange(0, 100, 1)

    for i, channel in enumerate(top_ten_accuracies_index):
        ax = axs[i]
        ax.plot(time, mean_accuracies[channel])
        ax.set_title('Channel #%i' %(channel + 1))
        ax.set_ylabel('Mean Accuracy')
        ax.set_xlabel('Time')
        ax.axvspan(mean_accuracies_max[channel,1],mean_accuracies_max[channel,1]+.1,color = 'red', alpha=0.5)

    plt.savefig(out_path + f'/Subject{sub}_top_ten_accuracies.png')
# %%
