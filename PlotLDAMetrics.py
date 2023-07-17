import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class PlotLDAMetrics(object):

    def __init__(
        self,
        estimator,
        setup_data,
        filter_channels = True
        ) :
        self.__estimator = estimator
        self.__elec_names = np.array(setup_data['elec_name'])
        self.__elec_areas = np.array(setup_data['elec_area'])

        self.__sort_scores(filter_channels)

    def __sort_scores(self, filter_channels:bool):

        mean_scores_max = np.zeros((self.__estimator._num_channels,3))

        for channel in range(self.__estimator._num_channels):
            mean_scores_max[channel, 0] = int(channel) # store the channel index
            mean_scores_max[channel, 1] = list(self.__estimator.mean_scores[channel]).index(max(self.__estimator.mean_scores[channel])) # the time point at which the maximum mean score occurs
            mean_scores_max[channel, 2] = max(self.__estimator.mean_scores[channel]) # value of the maximum mean score in a particular channel for all time points

        if filter_channels == True:
            elec_areas_filtered_idx = [i for i,ea in enumerate(self.__elec_areas) if ea not in ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid','lesion L','ventricle L','ventricle R']]
            sorted_indices = mean_scores_max[elec_areas_filtered_idx,2].argsort()[::-1]
            self.mean_scores_max_sorted = mean_scores_max[elec_areas_filtered_idx][sorted_indices]
            self.elec_names_sorted = self.__elec_names[np.int_(self.mean_scores_max_sorted[:, 0])]
            self.elec_areas_sorted = self.__elec_areas[np.int_(self.mean_scores_max_sorted[:, 0])]
            
        else:
            sorted_indices = mean_scores_max[:,2].argsort()[::-1]
            self.mean_scores_max_sorted = mean_scores_max[sorted_indices]
            self.elec_names_sorted = self.__elec_names[sorted_indices]
            self.elec_areas_sorted = self.__elec_areas[sorted_indices]

    def plot_sorted_scores(self, out_path:str):
        num_channels = len(self.mean_scores_max_sorted)
        
        fig, axs = plt.subplots(3, 1, figsize=(24,24), gridspec_kw={'height_ratios' : [1,1,1]})

        axs[0].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
        axs[0].set_ylabel('Peak Mean Score')
        axs[0].set_xlabel('Channels (from most to least accurate)')
        axs[0].plot(np.arange(0,num_channels,1), self.mean_scores_max_sorted[:,2])

        axs[1].set_title('Scatter Plot of Peak Score of LDA Models (from greatest to least) with error bars')
        axs[1].set_ylabel('Peak Mean Score')
        axs[1].set_xlabel('Channels (from most to least accurate)')
        axs[1].scatter(np.arange(0,num_channels,1), self.mean_scores_max_sorted[:,2])
        axs[1].errorbar(np.arange(0,num_channels,1), self.mean_scores_max_sorted[:,2], yerr=self.__estimator.std_scores[np.int_(self.mean_scores_max_sorted[:,0]),np.int_(self.mean_scores_max_sorted[:,1])].flatten(), fmt="o")
        
        axs[2].set_title('Time of Peak Score of LDA Models')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].set_xlabel('Channels (from most to least accurate)')
        time = self.mean_scores_max_sorted[:,1]/(20/self.__estimator._time_resolution) - 3
        axs[2].scatter(np.arange(0, num_channels), time)
        axs[2].axhline(y = 0, color = 'red', alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path + f'_sorted_scores')
        
        plt.figure(figsize=(24,24))
        plt.title('Sorted Peak Score of LDA Models (from greatest to least)')
        plt.ylabel('Channel Names')
        plt.xlabel('Score')
        plt.hlines(y=np.arange(0,num_channels), xmin=0.5, xmax=self.mean_scores_max_sorted[:,2][::-1],
            color='blue', alpha=0.6, linewidth=2)
        plt.yticks(np.arange(0,num_channels), labels=self.elec_areas_sorted[::-1])

        plt.savefig(out_path + f'_sorted_scores_hline')
    
    def plot_sorted_scores_per_channel(self, num_plots:int, out_path:str):
        time_resolution = self.__estimator._time_resolution
        rescaled_timesteps = self.__estimator._rescaled_timesteps
        
        times = (np.arange(0, rescaled_timesteps, 1) / (20/time_resolution)) - 3 # time 0 seconds denotes when the subject starts moving (i.e. 3 seconds into the data)

        fig, axs = plt.subplots(num_plots, 1, figsize=(24, 8 * num_plots))

        for i, trial_data in enumerate(self.mean_scores_max_sorted[:num_plots]):
            channel, time, peak_accuracy = trial_data
            time = time/(20/time_resolution) - 3
            ax = axs[i]
            ax.plot(times[:], self.__estimator.mean_scores[int(channel)])
            ax.set_title('Electrode %s in the %s' %(self.elec_names_sorted[i], self.elec_areas_sorted[i]))
            ax.set_ylabel('Score')
            ax.set_xlabel('Time (seconds)')
            # ax.tick_params(axis='both', labelsize=12)
            ax.axvline(time, color = 'red', alpha=0.5)
            ax.axvline(0, color = 'blue', alpha=0.5, ls = '--')
            ax.annotate(f'(Time: {time:.2f}s, Score: {peak_accuracy:.2f})', xy=(time + .05 ,.6), fontsize = 12)
        
        plt.savefig(out_path + f'_sorted_scores_per_channel')
        plt.show()

    def plot_power_heatmap(self, plot_metric, num_plots:int, out_path:str):
        num_freqs = self.__estimator._num_freqs
        time_resolution = self.__estimator._time_resolution
        rescaled_timesteps = self.__estimator._rescaled_timesteps

        if rescaled_timesteps >= 50:
            num_xticks = int(rescaled_timesteps/2)
        else:
            num_xticks = rescaled_timesteps

        xticks = np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)
        xticklabels = np.around(np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)/(20/time_resolution) - 3, decimals=2)

        if num_freqs == 5: 
            # sets the y-tick labels for EEG frequency bands
            yticks = np.arange(num_freqs)
            yticklabels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        else:
            # sets the y-tick labels for all frequencies in data
            yticks = np.arange(num_freqs, step=2)
            yticklabels = [round(i,1) for i in np.logspace(np.log2(2),np.log2(150), num = len(yticks),base=2, dtype=np.float16)]

        for i, trial_data in enumerate(self.mean_scores_max_sorted[:num_plots]):
            channel, time, peak_accuracy = trial_data

            low_bet_powers = self.__estimator.low_bet_avg_powers[int(channel)]
            high_bet_powers = self.__estimator.high_bet_avg_powers[int(channel)]
            diff_bet_powers = self.__estimator.diff_avg_powers[int(channel)]
            lda_coef = self.__estimator.lda_coefs[int(channel)]

            fig, axs = plt.subplots(2, 2,figsize=(24, 20))
            
            # Plot power per frequency as a function of time, power averaged across all respective trials (high or low bet trials) 
            sns.heatmap(high_bet_powers.T, ax=axs[0][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[0][0].set_title('Electrode %s in the %s \n High Bet Z-Scored Frequency Power (n = %s)' %(self.elec_names_sorted[i], self.elec_areas_sorted[i], "~"))
            axs[0][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            axs[0][0].set_xticks(xticks, labels = xticklabels)

            sns.heatmap(low_bet_powers.T, ax=axs[0][1], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[0][1].set_title('Electrode %s in the %s \n Low Bet Z-Scored Frequency Power (n = %s)' %(self.elec_names_sorted[i], self.elec_areas_sorted[i], "~"))
            axs[0][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            axs[0][1].set_xticks(xticks, labels = xticklabels)

            # ax = sns.heatmap(diff_bet_powers.T, vmin=-.4, vmax=.4, cbar_kws={"label": "Frequency Power", "pad": 0.1}, cmap='PRGn')
            # ax.set_title('Electrode %s in the %s \n Difference in Frequency Power (High - Low Bet)' %(elec_names[i], elec_areas[i]))
            # ax.set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            # ax.set_xticks(xticks, labels = xticklabels)
            # ax.axvline(time, color = 'red', alpha=0.5)
            # ax.axvline(0, color = 'blue', alpha=0.25, ls = '--')
            # ax_ = ax.twinx()
            # sns.lineplot(x=np.arange(0,rescaled_timesteps), y=plot_metric[int(channel)], color='blue', ax=ax_) # Make the overlayed metric an optional variable user can select
            # ax_.set_ylabel('Mean LDA Score')

            # Plot power per frequency at a particular timestep for each respective trial (high or low bet trials)
            # sns.heatmap(low_bet_powers[int(time)].T, ax=axs[0][0], vmin=-3, vmax=3, cbar_kws={"label": "Frequency Power"}, cmap='PRGn')
            # axs[0][0].set_title('Electrode %s in the %s at time %s \n Low Bet Frequency Power' %(elec_names[i], elec_areas[i], round(time/(20/time_resolution) - 3,2)))
            # axs[0][0].set(xlabel="Trial Indices", ylabel="Frequency (Hz)")

            # sns.heatmap(high_bet_powers[int(time)].T, ax=axs[0][1], vmin=-3, vmax=3, cbar_kws={"label": "Frequency Power"}, cmap='PRGn')
            # axs[0][1].set_title('Electrode %s in the %s at time %s \n High Bet Frequency Power' %(elec_names[i], elec_areas[i], round(time/(20/time_resolution) - 3,2)))
            # axs[0][1].set(xlabel="Trial Indices", ylabel="Frequency (Hz)")

            # Plots the difference in power frequency for high and low bet trials
            sns.heatmap(diff_bet_powers.T, ax=axs[1][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power", "pad": 0.1}, cmap='PRGn')
            axs[1][0].set_title('Electrode %s in the %s \n Difference in Z-Scored Frequency Power (High - Low Bet)' %(self.elec_names_sorted[i], self.elec_areas_sorted[i]))
            axs[1][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            axs[1][0].set_xticks(xticks, labels = xticklabels)
            ax = axs[1][0].twinx()
            sns.lineplot(x=np.arange(0,rescaled_timesteps), y=plot_metric[int(channel)].flatten(), color='blue', ax=ax) # Make the overlayed metric an optional variable user can select
            ax.set_ylabel('Mean LDA Score')
            axs[1][0].axvline(time, color = 'red', alpha=0.5)
            axs[1][0].axvline(0, color = 'blue', alpha=0.25, ls = '--')

            # Plots the LDA coefficients for each frequency band over time
            sns.heatmap(lda_coef.T, ax=axs[1][1], vmin=-1, vmax=1, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[1][1].set_title('LDA coefficient values for all frequencies \n at %s in %s' %(self.elec_names_sorted[i], self.elec_areas_sorted[i]))
            axs[1][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            axs[1][1].set_xticks(xticks, labels = xticklabels)

            for axs_ in axs:
                for ax in axs_:
                    # ax.set_xticks(xticks)
                    # ax.set_xticklabels(xticklabels)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels, rotation = 0)
                    ax.axes.invert_yaxis()
                    ax.axvline(time, color = 'red', alpha=0.5)
                    ax.axvline(12, color = 'blue', alpha=0.5, ls = '--')

                    for label in ax.xaxis.get_ticklabels()[1::2]:
                        label.set_visible(False)

            plt.savefig(out_path + '_heatmap_%s_%s'%(self.elec_names_sorted[i], self.elec_areas_sorted[i]))
            plt.show()