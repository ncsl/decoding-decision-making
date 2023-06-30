#%%
import h5py 
import mat73
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import mne
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_1samp

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
        'out_path_metrics': f'Metrics/Subject{sub}',
        'out_path_plots': f'Plots/Subject{sub}'
    }
## %%
# %%
def train_LDA_model(data, y, channel, time, time_resolution):
    num_timesteps = data.shape[3]

    # Checks that inputted value for time resolution is valid
    if not(num_timesteps % time_resolution == 0):
            raise Exception("Invalid time resolution size, num_timesteps % resolution > 0")
    
    X = data[:, channel, :, time:time+time_resolution].mean(2) 

    low_bet_avg_power = X[np.where(y == 0), :][0]
    high_bet_avg_power = X[np.where(y == 1), :][0]

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    cv_results = cross_validate(lda, X, y, cv = 5, return_estimator=True)

    estimators = cv_results['estimator']
    scores = cv_results['test_score']

    best_score = max(scores)
    best_index = list(scores).index(best_score)
    best_lda = estimators[best_index] # select the best performing model after cross-validation

    dval = np.dot(X, best_lda.coef_.T).T[0] + best_lda.intercept_ # calculate decision value 

    t_stat = ttest_1samp(dval, popmean=0).statistic # perform 1-sided t-test on decision values corresponding to high bet
    return best_score, dval, t_stat, low_bet_avg_power, high_bet_avg_power
## %%
# %%
def calculate_LDA_metrics(data, y, time_resolution):
    # Code to train LDA model for all channels and timepoints for Subject 6
    num_trials, num_channels, num_freqs, num_timesteps = data.shape
    rescaled_timesteps = int(num_timesteps/time_resolution)

    # A dictionary that stores all the metrics of the LDA model
    metrics = {
        'Best Scores' : np.zeros((num_channels, rescaled_timesteps)),
        'Decision Values' : np.zeros((num_channels, rescaled_timesteps, num_trials)),
        'T Stats' : np.zeros((num_channels,rescaled_timesteps)),
        'Low Bet Average Powers' : np.zeros((num_channels, rescaled_timesteps, len(y) - int(y.sum()), num_freqs)),
        'High Bet Average Powers' : np.zeros((num_channels, rescaled_timesteps, int(y.sum()), num_freqs)),
        'Time Resolution' : time_resolution
    }

    for channel in range(num_channels):
        for time in range(rescaled_timesteps):
            metrics['Best Scores'][channel, time], metrics['Decision Values'][:, time], metrics['T Stats'][channel, time], metrics['Low Bet Average Powers'][channel, time], metrics['High Bet Average Powers'][channel, time] = train_LDA_model(data=data, y=y, channel=channel, time=time*time_resolution, time_resolution=time_resolution)
    
    return metrics
## %%
# %%
def shuffle_y(y):
    # Get the locations for each particular card value
    card_value_indices = []
    for i in [2,4,6,8,10]:
        card_value_indices.append(np.where(subject_cards == i)[0])

    y_shuffled = np.zeros(y.shape)

    # Ensure that the number of high bets in the shuffled y labels is consistent with the card value
    for indices in card_value_indices:
        temp = indices
        num_high_bets = y[indices].sum() + round(np.random.uniform(-1,1)*y[indices].sum()*0.2) # Get the number of high bets for a particular card value and add some randomness to it
        for j in range(num_high_bets):
            if np.any(temp):
                # Pick a random location from all possible locations of that particular card value and set it to 1 (ie high bet)
                rand = np.random.choice(temp)
                y_shuffled[rand] = 1
                rand_index = np.where(temp == rand)[0]
                temp = np.delete(temp,rand_index) # Remove that location from being able to be chosen again
        y_shuffled[temp] = 0 # set all other locations for that particular card value to 0 (ie low bet)

    return y_shuffled
## %%
# %%
def get_shuffled_t_stats(data, y):
  np.random.seed()
  y_shuffled = shuffle_y(y)
  metrics = calculate_LDA_metrics(data=data, y=y_shuffled, time_resolution=2)
    
  return metrics['T Stats']
## %%
# %%
def plot_sorted_scores(metrics, best_scores_max_sorted, out_path):
    num_channels = data.shape[1]
    
    fig, axs = plt.subplots(3, 1, figsize=(24, 18))

    axs[0].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
    axs[0].set_ylabel('Peak Accuracy')
    axs[0].set_xlabel('Channels (from most to least accurate)')
    axs[0].plot(best_scores_max_sorted[:,2])

    axs[1].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
    axs[1].set_ylabel('Peak Accuracy')
    axs[1].set_xlabel('Channels (from most to least accurate)')
    axs[1].bar(np.arange(0,num_channels), best_scores_max_sorted[:,2])
    axs[1].set_ylim(min(best_scores_max_sorted[:,2]) - 0.025, max(best_scores_max_sorted[:,2]) + 0.025)

    axs[2].set_title('Time of Peak Score of LDA Models')
    axs[2].set_ylabel('Time (seconds)')
    axs[2].set_xlabel('Channels (from most to least accurate)')
    time = best_scores_max_sorted[:,1]/(20/metrics['Time Resolution']) -3
    axs[2].scatter(np.arange(0, num_channels), time)
    
    plt.savefig(out_path + f'_sorted_scores')
    plt.show()
## %%
# %%
def plot_sorted_scores_per_channel(metrics, best_scores_max_sorted, num_plots, out_path):
    num_timesteps = data.shape[3]

    time_resolution = metrics['Time Resolution']
    rescaled_timesteps = int(num_timesteps/time_resolution)
    times = (np.arange(0, rescaled_timesteps, 1) / (20/time_resolution)) - 3 # time 0 seconds denotes when the subject starts moving (i.e. 3 seconds into the data)

    fig, axs = plt.subplots(num_plots, 1, figsize=(24, 6 * num_plots))
    
    for i, trial_data in enumerate(best_scores_max_sorted[:num_plots]):
        channel, time, peak_accuracy = trial_data
        time = time/(20/time_resolution) - 3
        ax = axs[i]
        ax.plot(times[:], metrics['Best Scores'][int(channel)])
        ax.set_title('Electrode %s in the %s' %(elec_names[int(channel)], elec_areas[int(channel)]))
        ax.set_ylabel('Score')
        ax.set_xlabel('Time (seconds)')
        ax.axvspan(time - .0025 ,time + .0025, color = 'red', alpha=0.5)
        ax.annotate(f'(Time: {time:.2f}s, Score: {peak_accuracy:.2f})', xy=(time + .05 ,.6))
    
    plt.savefig(out_path + f'_sorted_scores_per_channel')
    plt.show()
## %%
# %%
def plot_power_heatmap(data, metrics, best_scores_max_sorted, num_plots, out_path):
    num_freqs, num_timesteps = data.shape[2:]
    time_resolution = metrics['Time Resolution']
    rescaled_timesteps = int(num_timesteps/time_resolution)

    num_x_ticks = 21
    num_y_ticks = 10

    yticklabels = np.logspace(np.log2(2),np.log2(150),num=num_y_ticks,base=2, dtype=np.int_)
    xticklabels = np.linspace(0, rescaled_timesteps, num=num_x_ticks, dtype=np.float16)/(20/time_resolution) - 3

    for i, elem in enumerate(xticklabels):
        xticklabels[i] = round(elem,1)

    yticks = np.linspace(0, num_freqs, num_y_ticks)
    xticks = np.linspace(0, num_timesteps, num_x_ticks)


    for i, trial_data in enumerate(best_scores_max_sorted[:num_plots]):
        channel, time, peak_accuracy = trial_data

        low_bet_powers = metrics['Low Bet Average Powers'][int(channel), int(time)]
        high_bet_powers = metrics['High Bet Average Powers'][int(channel), int(time)]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24, 8))
        sns.heatmap(low_bet_powers.T, ax=ax1, vmin=-5, vmax=5)
        ax1.set_title('Electrode %s in the %s at time %s: Low Bet Frequency Power' %(elec_names[int(channel)], elec_areas[int(channel)], round(time/(20/time_resolution) - 3,2)))

        sns.heatmap(high_bet_powers.T, ax=ax2, vmin=-5, vmax=5)
        ax2.set_title('Electrode %s in the %s at time %s: High Bet Frequency Power' %(elec_names[int(channel)], elec_areas[int(channel)], round(time/(20/time_resolution) - 3,2)))

        for ax in (ax1,ax2):
            ax.axes.invert_yaxis()
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set(xlabel="Trial Indices", ylabel="Frequency (Hz)")
        
        plt.savefig(out_path + f'_heatmap_{i}')
        plt.show()
## %%
# %%
def sort_scores(data, metrics):
    num_channels = data.shape[1]
    best_scores_max = np.zeros((num_channels,3))

    for channel in range(num_channels):
        best_scores_max[channel, 0] = channel # store the channel index
        best_scores_max[channel, 1] = list(metrics['Best Scores'][channel]).index(max(metrics['Best Scores'][channel])) # the time point at which the maximum accuracy occurs
        best_scores_max[channel, 2] = max(metrics['Best Scores'][channel]) # value of the max score in a particular channel for all time points

    sorted_indices = best_scores_max[:,2].argsort()[::-1]

    best_scores_max_sorted = best_scores_max[sorted_indices]
    elec_names_sorted = elec_names[sorted_indices]
    elec_areas_sorted = elec_areas[sorted_indices]
    
    return best_scores_max_sorted, elec_names_sorted, elec_areas_sorted
## %%
# %%
def plot_scores(data, metrics, out_path_plots):    
    best_scores_max_sorted, elec_names_sorted, elec_areas_sorted = sort_scores(data, metrics)
    plot_sorted_scores(metrics, best_scores_max_sorted, out_path_plots)
    plot_sorted_scores_per_channel(metrics, best_scores_max_sorted, 10, out_path_plots)
    plot_power_heatmap(data, metrics, best_scores_max_sorted, 10, out_path_plots)
## %%
#%%
for sub in subs:
    # load appropriate files/data
    raw_file = h5py.File(file_paths[sub]['raw_path'])
    setup_data = mat73.loadmat(file_paths[sub]['setup_path'])

    out_path_plots = file_paths[sub]['out_path_plots']
    out_path_metrics = file_paths[sub]['out_path_metrics']

    # instantiate approparite variables  
    bets = setup_data['filters']['bets']
    good_trials = np.where(np.isnan(bets) == False)[0]
    bets = bets[good_trials]
    subject_cards = setup_data['filters']['card1'][good_trials] # get the subject's card values for the good trials

    elec_names = np.array(setup_data['elec_name'])
    elec_areas = np.array(setup_data['elec_area'])

    data = np.load(file_paths[sub]['data_path'])
    y = np.asarray([(0 if bet == 5 else 1) for bet in bets]) # 0 = low bet ($5), 1 = high bet ($20)

    # calculate metrics and create plots
    metrics = calculate_LDA_metrics(data,y,time_resolution=2)

    with open(out_path_metrics+'_LDA_metrics.csv', 'w', newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    best_scores_max_sorted, elec_names_sorted, elec_areas_sorted = sort_scores(data,metrics)
    
    sorted_scores = {
        'Sorted Max Scores' : best_scores_max_sorted,
        'Sorted Electrode Names' : elec_names_sorted,
        'Sorted Electrode Areas' : elec_areas_sorted
    }

    with open(out_path_metrics+'_sorted_scores.csv', 'w', newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted_scores.keys())
        writer.writeheader()
        writer.writerow(sorted_scores)
    
    plot_scores(data, metrics, out_path_plots)
## %%
# %%
