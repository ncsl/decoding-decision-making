import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class PerChannelTimestep(object):
    """Visualizes model performance for LDA models trained on each channel and timestep of the data."""

    def __init__(
        self,
        estimator,
        filter_channels = False
        ) :
        self.__estimator = estimator

        self._sort_scores(filter_channels)

    def _sort_scores(self, filter_channels:bool):
        """Sort channels from greatest to least maximum LDA scores, and indicate timepoint at which maximum LDA score occurs."""
        max_mean_scores = np.zeros((self.__estimator.num_channels,3))

        for channel in range(self.__estimator.num_channels):
            max_mean_scores[channel, 0] = int(channel) # store the channel index
            max_mean_scores[channel, 1] = list(self.__estimator.mean_scores[channel]).index(max(self.__estimator.mean_scores[channel])) # the time point at which the maximum mean score occurs
            max_mean_scores[channel, 2] = max(self.__estimator.mean_scores[channel]) # value of the maximum mean score in a particular channel for all time points

        if filter_channels == True:
            elec_areas_filtered_idx = [i for i,ea in enumerate(self.__estimator.elec_areas) if ea not in ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid','lesion L','ventricle L','ventricle R']]
            sorted_indices = max_mean_scores[elec_areas_filtered_idx,2].argsort()[::-1]
            self.sorted_max_mean_scores = max_mean_scores[elec_areas_filtered_idx][sorted_indices]
            self.sorted_elec_names = [self.__estimator.elec_names[i] for i in np.int_(self.sorted_max_mean_scores[:, 0])]
            self.sorted_elec_areas = [self.__estimator.elec_areas[i] for i in np.int_(self.sorted_max_mean_scores[:, 0])]
            
        else:
            sorted_indices = max_mean_scores[:,2].argsort()[::-1]
            self.sorted_max_mean_scores = max_mean_scores[sorted_indices]
            self.sorted_elec_names = [self.__estimator.elec_names[i] for i in sorted_indices]
            self.sorted_elec_areas = [self.__estimator.elec_areas[i] for i in sorted_indices]

    def plot_sorted_scores(self, out_path:str):
        """Visualize the LDA model scores (sorted from greatest to least) for all channels."""
        num_channels = len(self.sorted_max_mean_scores)
        
        fig, axs = plt.subplots(3, 1, figsize=(24,24), gridspec_kw={'height_ratios' : [1,1,1]})

        axs[0].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
        axs[0].set_ylabel('Peak Mean Score')
        axs[0].set_xlabel('Channels (from most to least accurate)')
        axs[0].plot(np.arange(0,num_channels,1), self.sorted_max_mean_scores[:,2])

        axs[1].set_title('Scatter Plot of Peak Score of LDA Models (from greatest to least) with error bars')
        axs[1].set_ylabel('Peak Mean Score')
        axs[1].set_xlabel('Channels (from most to least accurate)')
        axs[1].scatter(np.arange(0,num_channels,1), self.sorted_max_mean_scores[:,2])
        axs[1].errorbar(np.arange(0,num_channels,1), self.sorted_max_mean_scores[:,2], yerr=self.__estimator.std_scores[np.int_(self.sorted_max_mean_scores[:,0]),np.int_(self.sorted_max_mean_scores[:,1])].flatten(), fmt="o")
        
        axs[2].set_title('Time of Peak Score of LDA Models')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].set_xlabel('Channels (from most to least accurate)')
        time = self.sorted_max_mean_scores[:,1]/(20/self.__estimator.time_resolution) - 3
        axs[2].scatter(np.arange(0, num_channels), time)
        axs[2].axhline(y = 0, color = 'red', alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path + f'_sorted_scores')
        
        plt.figure(figsize=(24,24))
        plt.title('Sorted Peak Score of LDA Models (from greatest to least)')
        plt.ylabel('Channel Names')
        plt.xlabel('Score')
        plt.hlines(y=np.arange(0,num_channels), xmin=0.5, xmax=self.sorted_max_mean_scores[:,2][::-1],
            color='blue', alpha=0.6, linewidth=2)
        plt.yticks(np.arange(0,num_channels), labels=self.sorted_elec_areas[::-1])

        plt.savefig(out_path + f'_sorted_scores_hline')
    
    def plot_sorted_scores_per_channel(self, num_plots:int, out_path:str):
        """Visualize the LDA model scores over time for top performing channels."""
        time_resolution = self.__estimator.time_resolution
        rescaled_timesteps = self.__estimator.rescaled_timesteps
        
        times = (np.arange(0, rescaled_timesteps, 1) / (20/time_resolution)) - 3 # time 0 seconds denotes when the subject starts moving (i.e. 3 seconds into the data)

        fig, axs = plt.subplots(num_plots, 1, figsize=(24, 8 * num_plots))

        for i, trial_data in enumerate(self.sorted_max_mean_scores[:num_plots]):
            channel, time, peak_accuracy = trial_data
            time = time/(20/time_resolution) - 3
            ax = axs[i]
            ax.plot(times[:], self.__estimator.mean_scores[int(channel)])
            ax.set_title('Electrode %s in the %s' %(self.sorted_elec_names[i], self.sorted_elec_areas[i]))
            ax.set_ylabel('Score')
            ax.set_xlabel('Time (seconds)')
            # ax.tick_params(axis='both', labelsize=12)
            ax.axvline(time, color = 'red', alpha=0.5)
            ax.axvline(0, color = 'blue', alpha=0.5, ls = '--')
            ax.annotate(f'(Time: {time:.2f}s, Score: {peak_accuracy:.2f})', xy=(time + .05 ,.6), fontsize = 12)
        
        plt.savefig(out_path + f'_sorted_scores_per_channel')
        plt.show()

    def plot_power_heatmap(self, plot_metric, num_plots:int, out_path:str):
        """Plot heatmaps of high/low frequency powers, difference in high/low frequency powers, and LDA coefficients."""
        num_freqs = self.__estimator.num_freqs
        time_resolution = self.__estimator.time_resolution
        rescaled_timesteps = self.__estimator.rescaled_timesteps

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

        for i, trial_data in enumerate(self.sorted_max_mean_scores[:num_plots]):
            channel, time, peak_accuracy = trial_data

            low_bet_powers = self.__estimator.low_bet_avg_powers[int(channel)]
            high_bet_powers = self.__estimator.high_bet_avg_powers[int(channel)]
            diff_bet_powers = self.__estimator.diff_avg_powers[int(channel)]
            lda_coef = self.__estimator.lda_coefs[int(channel)]

            fig, axs = plt.subplots(2, 2,figsize=(24, 20))
            
            # Plot power per frequency as a function of time, power averaged across all respective trials (high or low bet trials) 
            sns.heatmap(high_bet_powers.T, ax=axs[0][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[0][0].set_title('Electrode %s in the %s \n High Bet Z-Scored Frequency Power (n = %s)' %(self.sorted_elec_names[i], self.sorted_elec_areas[i], "~"))
            axs[0][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            # axs[0][0].set_xticks(xticks, labels = xticklabels)

            sns.heatmap(low_bet_powers.T, ax=axs[0][1], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[0][1].set_title('Electrode %s in the %s \n Low Bet Z-Scored Frequency Power (n = %s)' %(self.sorted_elec_names[i], self.sorted_elec_areas[i], "~"))
            axs[0][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            # axs[0][1].set_xticks(xticks, labels = xticklabels)

            # Plot power per frequency at a particular timestep for each respective trial (high or low bet trials)
            # sns.heatmap(low_bet_powers[int(time)].T, ax=axs[0][0], vmin=-3, vmax=3, cbar_kws={"label": "Frequency Power"}, cmap='PRGn')
            # axs[0][0].set_title('Electrode %s in the %s at time %s \n Low Bet Frequency Power' %(elec_names[i], elec_areas[i], round(time/(20/time_resolution) - 3,2)))
            # axs[0][0].set(xlabel="Trial Indices", ylabel="Frequency (Hz)")

            # sns.heatmap(high_bet_powers[int(time)].T, ax=axs[0][1], vmin=-3, vmax=3, cbar_kws={"label": "Frequency Power"}, cmap='PRGn')
            # axs[0][1].set_title('Electrode %s in the %s at time %s \n High Bet Frequency Power' %(elec_names[i], elec_areas[i], round(time/(20/time_resolution) - 3,2)))
            # axs[0][1].set(xlabel="Trial Indices", ylabel="Frequency (Hz)")

            # Plots the difference in power frequency for high and low bet trials
            sns.heatmap(diff_bet_powers.T, ax=axs[1][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power", "pad": 0.1}, cmap='PRGn')
            axs[1][0].set_title('Electrode %s in the %s \n Difference in Z-Scored Frequency Power (High - Low Bet)' %(self.sorted_elec_names[i], self.sorted_elec_areas[i]))
            axs[1][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            # axs[1][0].set_xticks(xticks, labels = xticklabels)
            ax = axs[1][0].twinx()
            sns.lineplot(x=np.arange(0,rescaled_timesteps), y=plot_metric[int(channel)].flatten(), color='blue', ax=ax) # Make the overlayed metric an optional variable user can select
            ax.set_ylabel('Mean LDA Score')
            axs[1][0].axvline(time, color = 'red', alpha=0.5)
            axs[1][0].axvline(0, color = 'blue', alpha=0.25, ls = '--')

            # Plots the LDA coefficients for each frequency band over time
            sns.heatmap(lda_coef.T, ax=axs[1][1], vmin=-1, vmax=1, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
            axs[1][1].set_title('LDA coefficient values for all frequencies \n at %s in %s' %(self.sorted_elec_names[i], self.sorted_elec_areas[i]))
            axs[1][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
            # axs[1][1].set_xticks(xticks, labels = xticklabels)

            for axs_ in axs:
                for ax in axs_:
                    ax.set_xticks(xticks, labels = xticklabels)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels, rotation = 0)
                    ax.axes.invert_yaxis()
                    ax.axvline(time, color = 'red', alpha=0.5)
                    ax.axvline(12, color = 'blue', alpha=0.5, ls = '--')

                    for label in ax.xaxis.get_ticklabels()[1::2]:
                        label.set_visible(False)

            plt.savefig(out_path + '_heatmap_%s_%s'%(self.sorted_elec_names[i], self.sorted_elec_areas[i]))
            plt.show()

class PerTimestepAllChannels(object):

    """Visualizes model performance for LDA models trained on all channels at each timestep of the data."""

    def __init__(
        self,
        estimator
        ) :
        self.__estimator = estimator

        self._sort_scores()
        self._convert_timesteps_to_time(3)
        
    def _sort_scores(self):
        """Sort the LDA scores from greatest to least, with indexes of channels saved."""
        enumerated_mean_scores = np.array(list(enumerate(self.__estimator.mean_scores.flatten())))
        sorted_indices = enumerated_mean_scores[:,1].argsort()[::-1]
        self.sorted_mean_scores = enumerated_mean_scores[sorted_indices]

    def _convert_timesteps_to_time(self, event_delay):
        """Convert timesteps into seconds while specifying timepoint 0."""
        self.__event_delay = event_delay
        # time point "0 seconds" denoted by event_delay
        self.__times = (np.arange(0, self.__estimator.rescaled_timesteps) / (20/self.__estimator.time_resolution)) - event_delay 

    
    def plot_sorted_scores(self, out_path:str):
        """Visualize the LDA model scores (sorted from greatest to least) for all timepoints."""
        num_timesteps = self.__estimator.rescaled_timesteps
        xticks = np.arange(0,num_timesteps,1)
        
        fig, axs = plt.subplots(3, 1, figsize=(24,24), gridspec_kw={'height_ratios' : [1,1,1]})

        axs[0].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
        axs[0].set_ylabel('Mean Score')
        axs[0].set_xlabel('Times (from most to least accurate)')
        axs[0].set_xticks(xticks, labels = self.__times[np.int_(self.sorted_mean_scores[:,0])])
        axs[0].plot(xticks ,self.sorted_mean_scores[:,1])

        axs[1].set_title('Scatter Plot of Peak Score of LDA Models (from greatest to least) with error bars')
        axs[1].set_ylabel('Mean Score')
        axs[1].set_xlabel('Times (from most to least accurate)')
        axs[1].set_xticks(xticks, labels = self.__times[np.int_(self.sorted_mean_scores[:,0])])
        axs[1].scatter(xticks, self.sorted_mean_scores[:,1])
        axs[1].errorbar(xticks, self.sorted_mean_scores[:,1], yerr=self.__estimator.std_scores[np.int_(self.sorted_mean_scores[:,0])].flatten(), fmt="o")

        axs[2].set_title('Sorted Peak Score of LDA Models (from greatest to least)')
        axs[2].set_ylabel('Times (from most to least accurate)')
        axs[2].set_xlabel('Score')
        axs[2].hlines(y=np.arange(0,num_timesteps), xmin=0.5, xmax=self.sorted_mean_scores[:,1][::-1],
            color='blue', alpha=0.6, linewidth=2)
        axs[2].set_yticks(np.arange(0,num_timesteps), labels=self.__times[np.int_(self.sorted_mean_scores[:,0])][::-1])

        fig.tight_layout()
        fig.savefig(out_path + '_sorted_scores_per_timestep_all_channels')
    
    def plot_power_heatmap(self, out_path:str):
        """Plot heatmaps of high/low frequency powers, difference in high/low frequency powers, and LDA coefficients."""
        num_freqs = self.__estimator.num_freqs
        time_resolution = self.__estimator.time_resolution
        rescaled_timesteps = self.__estimator.rescaled_timesteps

        low_bet_powers = self.__estimator.low_bet_avg_powers
        high_bet_powers = self.__estimator.high_bet_avg_powers
        diff_bet_powers = self.__estimator.diff_avg_powers
        lda_coef = self.__estimator.lda_coefs

        if rescaled_timesteps >= 50:
            num_xticks = int(rescaled_timesteps/2)
        else:
            num_xticks = rescaled_timesteps

        if num_freqs == 5: 
            # sets the y-ticks when using EEG frequency bands
            yticks = np.arange(diff_bet_powers.shape[1], step=5)
        else:
            # sets the y-ticks when using all frequencies in data
            yticks = np.arange(diff_bet_powers.shape[1], step=63)
            # yticklabels = [round(i,1) for i in np.logspace(np.log2(2),np.log2(150), num = len(yticks),base=2, dtype=np.float16)]

        xticks = np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)
        xticklabels = np.around(np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)/(20/time_resolution) - self.__event_delay, decimals=2)

        fig, axs = plt.subplots(2, 2,figsize=(24, 20))
        
        # Plot power per frequency as a function of time, power averaged across all respective trials (high or low bet trials) 
        sns.heatmap(high_bet_powers.T, ax=axs[0][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
        axs[0][0].set_title('High Bet Z-Scored Frequency Power')
        # axs[0][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")

        sns.heatmap(low_bet_powers.T, ax=axs[0][1], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
        axs[0][1].set_title('Low Bet Z-Scored Frequency Power')
        # axs[0][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")

        # Plots the difference in power frequency for high and low bet trials
        sns.heatmap(diff_bet_powers.T, ax=axs[1][0], vmin=-.4, vmax=.4, cbar_kws={"label": "Z-Scored Frequency Power", "pad": 0.1}, cmap='PRGn')
        axs[1][0].set_title('Difference in Z-Scored Frequency Power (High - Low Bet)')
        # axs[1][0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
        ax = axs[1][0].twinx()
        sns.lineplot(x=np.arange(0,rescaled_timesteps), y=self.__estimator.mean_scores.flatten(), color='blue', ax=ax) # Make the overlayed metric an optional variable user can select
        ax.set_ylabel('Mean LDA Score')

        # Plots the LDA coefficients for each frequency band over time
        sns.heatmap(lda_coef.T, ax=axs[1][1], vmin=-1, vmax=1, cbar_kws={"label": "Z-Scored Frequency Power"}, cmap='PRGn')
        axs[1][1].set_title('LDA coefficient values (on channels and frequencies)')
        # axs[1][1].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")

        for axs_ in axs:
            for ax in axs_:
                ax.set_xticks(xticks, labels = xticklabels, rotation = 90)
                ax.set_yticks(yticks)
        #        ax.set_yticklabels(yticklabels, rotation = 0)
                ax.set(xlabel="Time (seconds)", ylabel="Frequency Bands * Channels")
                ax.axes.invert_yaxis()
                ax.axvline(self.sorted_mean_scores[0,0], color = 'red', alpha=0.5)
                ax.axvline(12, color = 'blue', alpha=1, ls = '--')

                for label in ax.xaxis.get_ticklabels()[1::2]:
                    label.set_visible(False)

        plt.savefig(out_path + '_heatmap_per_timestep_all_channels')
        plt.show()

    def plot_contributing_channels(self, alpha):
        """Visualize the heatmap of the LDA coefficients for top channels with mean LDA coefficient absolute values greater than alpha."""
        num_freqs = self.__estimator.num_freqs
        time_resolution = self.__estimator.time_resolution
        rescaled_timesteps = self.__estimator.rescaled_timesteps

        if rescaled_timesteps >= 50:
            num_xticks = int(rescaled_timesteps/2)
        else:
            num_xticks = rescaled_timesteps
        
        yticks = np.arange(num_freqs)
        yticklabels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

        xticks = np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)
        xticklabels = np.around(np.linspace(0, rescaled_timesteps - 1, num=num_xticks, dtype=np.int_)/(20/time_resolution) - self.__event_delay, decimals=2)


        avg_coefs = np.abs(self.__estimator.lda_coefs).mean(0)
        idxs = []

        for i in avg_coefs.argsort()[::-1]:
            if avg_coefs[i] > alpha:
                idxs.append(i)
            else:
                break
        
        channels = [int(i/self.__estimator.num_freqs) for i in idxs]

        fig, axs = plt.subplots(len(idxs), 1, figsize=(12,7*len(idxs)))

        for i, ch in enumerate(channels):
            ch_idx = ch*self.__estimator.num_freqs
            sns.heatmap(self.__estimator.lda_coefs.T[ch_idx:ch_idx+self.__estimator.num_freqs], ax=axs[i], cmap='PRGn')

            axs[i].set_xticks(xticks, labels = xticklabels, rotation = 90)
            axs[i].set_xlabel('Time (seconds)')
            axs[i].set_yticklabels(yticklabels, rotation = 0)
            axs[i].set_ylabel('Frequency Band')
            axs[i].axes.invert_yaxis()
            axs[i].axvline(self.sorted_mean_scores[0,0], color = 'red', alpha=0.5)
            axs[i].axvline(12, color = 'blue', alpha=1, ls = '--')
            axs[i].set_title('Electrode %s in the %s \n LDA Coefficients' %(self.__estimator.elec_names[ch], self.__estimator.elec_areas[ch]))

        plt.tight_layout()