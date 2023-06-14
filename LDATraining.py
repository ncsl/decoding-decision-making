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

# %%
def train_LDA_model(data, y, channel, time):
    X = data[:, channel, :, time] # get the EEG data for a particular channel and time point

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    cv_results = cross_validate(lda, X, y, cv=5, return_estimator=True)
    
    model_accuracies[channel, time] = cv_results['test_score'] # store the accuracies of the cross-validated model
    max_index = list(model_accuracies[channel,time]).index(model_accuracies[channel,time].max()) 
    best_lda = cv_results['estimator'][max_index] # select the best performing model after cross-validation
    
    prob_values[channel,time] = best_lda.predict_log_proba(X) # calculate the log of probabilities for classification of each class (ie decision value for each class)
    t_values[channel,time] = stats.ttest_ind(prob_values[channel,time,:,0],prob_values[channel,time,:,1]).statistic # perform t-test on log prob values (ie decision values for each class)
## %%

# %%
def shuffle_y(y, alpha):
    # Get the indices for each of the card values in subject_cards
    card_value_indices = []
    for i in [2,4,6,8,10]:
        card_value_indices.append(np.where(subject_cards == i)[0])

    y_shuffled = np.zeros(y.shape)
    for indices in card_value_indices:
        temp = indices
        num_high_bets = y[indices].sum() + round(np.random.uniform(-1,1)*y[indices].sum()*alpha) # (1-alpha) is how much 
        for j in range(num_high_bets):
            if np.any(temp):
                rand = np.random.choice(temp)
                y_shuffled[rand] = 1
                rand_index = np.where(temp == rand)[0]
                temp = np.delete(temp,rand_index)
        y_shuffled[temp] = 0

    return y_shuffled
## %%

# %%
def plot_LDA_accuracies():
    # Calculate the mean accuracies for the LDA model after cross-validation
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
        ax.annotate(f'(Time: {mean_accuracies_max[channel,1]:.0f}, Accuracy: {mean_accuracies_max[channel,0]:.2f})', xy=(mean_accuracies_max[channel,1] + 1,.5))

    plt.savefig(out_path_graphs + f'/Subject{sub}_top_ten_accuracies.png')
## %%

# %%

# subs = ['06','07','10','12','13','15','16','17','18','21']
subs = ['06', '07']
file_paths = {}

for sub in subs:
    # create a dictionary holding the file paths
    ncsl_share = '/mnt/ncsl_share'
    file_paths[sub] = {
        'setup_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat',
        'raw_path': ncsl_share + f'/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat',
        'data_path': ncsl_share + f'/Daniel/Data/Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy',
        'out_path_graphs': 'Top_Ten_Accuracy_Graphs',
        'out_path_tvalues': f't_values'
    }
## %%

#%%
for sub in subs:
    # load appropriate files/data
    raw_file = h5py.File(file_paths[sub]['raw_path'])
    setup_data = mat73.loadmat(file_paths[sub]['setup_path'])
    data = np.load(file_paths[sub]['data_path'])
    out_path_graphs = file_paths[sub]['out_path_graphs']
    out_path_tvalues = file_paths[sub]['out_path_tvalues']

    # instantiate approparite variables
    num_trials, num_channels, num_freqs, num_timesteps = data.shape
  
    bets = setup_data['filters']['bets']
    good_trials = np.where(np.isnan(bets) == False)[0]
    bets = bets[good_trials]
    subject_cards = setup_data['filters']['card1'][good_trials] # get the subject's card values for the good trials

    model_accuracies = np.zeros((num_channels, num_timesteps, 5)) 
    prob_values = np.zeros((num_channels, num_timesteps, num_trials, 2))
    t_values = np.zeros((num_channels,num_timesteps))

    # Trains an LDA model on preprocessed data, implements cross validation, and performs t-test on decision values  
    y = np.asarray([(0 if bet == 5 else 1) for bet in bets]) # 0 = low bet ($5), 1 = high bet ($20)
    for channel in range(num_channels):
        for time in range(num_timesteps):
            train_LDA_model(data, y, channel, time)

    np.save(f'{out_path_tvalues}/Subject{sub}_tvalues.npy',t_values) # save t-values

    for i in range(100):
    y_shuffled = shuffle_y(y,0.2)
    for channel in range(num_channels):
        for time in range(num_timesteps):
            train_LDA_model(data, y_shuffled, channel, time)
    
    np.save(f'{out_path_tvalues}/Subject{sub}_shuffled{i}_tvalues.npy',t_values) # save t-values

## %%