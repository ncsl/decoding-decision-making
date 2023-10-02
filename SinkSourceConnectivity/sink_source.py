import mne
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

class SinkSource():
    def __init__(self, raw_file, setup_data):
        self.Fs = raw_file['Fs'][0][0]

        self._setup_data = setup_data
        self._elec_names = setup_data['elec_name']
        self._elec_areas = setup_data['elec_area']
        self._filter_channels()

    def _filter_channels(self):
        self._filtered_elec_areas_idxs = [i for i,ea in enumerate(self._elec_areas) if ea not in 
                            ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid',
                                'lesion L','ventricle L','ventricle R']]
        self._filtered_elec_areas = [self._elec_areas[i] for i in self._filtered_elec_areas_idxs]
        self._filtered_elec_names = [self._elec_names[i] for i in self._filtered_elec_areas_idxs]
        filtered_num_channels = len(self._filtered_elec_areas_idxs)

    def _get_unique_channels(self):
        unique_ch_areas = np.unique(self._filtered_elec_areas)

        ch_area_idxs = []

        for ch_area in unique_ch_areas:
            ch_area_idxs.append(np.where(np.isin(self._filtered_elec_areas, ch_area)))

        return unique_ch_areas, ch_area_idxs


    def _laplacian_reference(self, raw_file):
        lfp_data = raw_file['lfpdata']
        lfp_all = lfp_data[:,:]
    
        # Filter Data Bandpass .5-200 Hz 
        filt = mne.filter.filter_data(lfp_all,self.Fs,0.5,200,method="iir")
        # notch filter 60 hz harmonics
        for notchfreq in [60,120,180]:
            filt = mne.filter.notch_filter(filt,self.Fs,notchfreq, method="iir")
        # decimate to 500 Hz 
        decFactor = int(self.Fs/500)
        filt = filt[:,::decFactor]

        ## For each channel in elec_names, get its index position in array, whether its on the end of the electrode shaft, and its neighboring indices 
        lap_ref_data = np.zeros(filt.shape)
        for i,en in enumerate(self._elec_names):
            if en in ["REF1","REF2","E","CZ","FZ","PZ"]:
                lap_ref_data[i,:] = filt[i,:]
                continue
            pattern = r"([a-z']+) *(\d+)"
            shaft_name = re.findall(pattern,en,re.IGNORECASE)[0][0]
            elec_num = re.findall(pattern,en,re.IGNORECASE)[0][1]
            en_plus1 = f"{shaft_name}{str(int(elec_num)+1)}"
            en_minus1 = f"{shaft_name}{str(int(elec_num)-1)}"
            if en_minus1 not in self._elec_names:
                neighbor_inds=[i-1]
            elif en_plus1 not in self._elec_names:
                neighbor_inds=[i+1]
            else:
                neighbor_inds = [ i-1,i+1]
            print(en, i, neighbor_inds,[self._elec_names[n] for n in neighbor_inds])
            neighbor_mean = np.mean(filt[neighbor_inds,:],axis=0)
            lap_ref_data[i,:] = filt[i,:] - neighbor_mean
        
        return lap_ref_data

    def _snapshot_data(self, raw_file, event:int, time_interval:list):
        lap_ref_data = self._laplacian_reference(raw_file)
        dsFs = 500

        good_trials = self._setup_data['filters']['trial'][self._setup_data['filters']['success']].astype(int)-1
        num_trials = len(good_trials)
        
        window_length = np.abs(time_interval[0])+ np.abs(time_interval[1])

        snapshot_data = np.zeros((num_trials,lap_ref_data.shape[0],int(window_length*dsFs)))

        # Snapshot around designated event
        for i,t in enumerate(good_trials):
            event_time = self._setup_data['trial_times'][t][0][self._setup_data['trial_words'][t][0]==event][0]
            #print(f'start move time = {start_move_time} for trial {t}')
            
            ## To go from the time to the index position in the lfp array, multiply time by Fs 
            # start_move_index = int(start_move_time*dsFs)
            #print(f'start move index = {start_move_index} for trial {t}')
            start_index = int((event_time + time_interval[0])*dsFs)
            end_index = start_index+int(window_length*dsFs)#int((start_move_time + 2.0)*Fs)
            data_slice = lap_ref_data[:,start_index:end_index]
            snapshot_data[i,:,:] = data_slice

        return snapshot_data
    
    def get_data(self, raw_file, event:int, time_interval:list):
        snapshot_data = self._snapshot_data(raw_file, event, time_interval)
        return snapshot_data[:,self._filtered_elec_areas_idxs,:]
    
    def get_y(self):
        unfiltered_bets = self._setup_data['filters']['bets']

        good_trials = np.where(np.isnan(unfiltered_bets) == False)[0] # extract indices of trials without the 'nan'
        bets = unfiltered_bets[good_trials] # get the bet values for the good trials

        return np.asarray([(0 if bet == 5 else 1) for bet in bets])
    
    def _estimateA(self, X):
        # Jeff Craley's Method using definition of least squares
        Y = X[:, 1:]
        Z = X[:, 0:-1]
        A_hat = Y @ np.linalg.pinv(Z)
        return A_hat
    
    def _estimateA_trial(self, data, fs=500, winsize=0.5):
        window = int(np.floor(winsize * fs))
        time = data.shape[1]
        n_chs = data.shape[0]
        n_wins = int(np.round(time / window))
        A_hat = np.zeros((n_chs, n_chs, n_wins))
        for win in range(0,n_wins):
            if win*window < data.shape[1]:
                data_win = data[:,win*window:(win+1)*window]
                A_hat[:,:,win] = self._estimateA(data_win)
                if win % 1000 == 0:
                    print(f"{win}/{n_wins} is computed")
        return A_hat
    
    def estimateA_all_trials(self, data):
        n_trials = data.shape[0]
        A_hat_all = []
        for trial in range(0,n_trials):
            A_hat_all.append(self._estimateA_trial(data[trial]))

        A_hat_all = np.asarray(A_hat_all)

        self.nWin = A_hat_all.shape[3]
        self.nCh = A_hat_all.shape[1]

        return A_hat_all
    
    def _identifySS(self, A):
        nCh = A.shape[0]

        A_abs = np.abs(A)
        A_abs[np.diag_indices_from(A)] = 0# set diagonals to zero

        # Compute row and column sums
        sum_A_r = np.sum(A_abs,axis=1)
        sum_A_c = np.sum(A_abs,axis=0)

        # Identify sources/sinks
        # Rank the channels from lowest (rank 1) to highest (rank nCh) based on row sum. Rank the channels from highest (rank 1) to
        # lowest (rank nCh) based on column sum. Sum the two ranks. Sinks = high rank sum and sources = low rank sum
        sort_ch_r = np.argsort(sum_A_r) # ascending
        row_ranks = np.argsort(sort_ch_r)  # rearrange the sorted channels back to being from 1:nCh
        row_ranks = row_ranks / nCh

        sort_ch_c = np.argsort(sum_A_c)[::-1] # descending
        col_ranks = np.argsort(sort_ch_c)   # rearrange the sorted channels back to being from 1:nCh
        col_ranks = col_ranks / nCh

        SI = np.sqrt(2) - np.sqrt((1-row_ranks)**2+(1-col_ranks)**2)

        return SI, row_ranks, col_ranks
    
    def computeSS(self, A_hat_all):
        A_win_all_trials = []
        SI_wins_all_trials = []
        row_ranks_all_trials = []
        col_ranks_all_trials = []

        for A_hat in A_hat_all:
            SI_wins = np.zeros((self.nCh, self.nWin))
            row_ranks = np.zeros((self.nCh, self.nWin))
            col_ranks = np.zeros((self.nCh, self.nWin))
            
            for iW in range(0,self.nWin):
                A_win = A_hat[:,:,iW]
                SI_wins[:, iW], row_ranks[:, iW], col_ranks[:, iW] = self._identifySS(A_win)

            A_win_all_trials.append(A_win)
            SI_wins_all_trials.append(SI_wins)
            row_ranks_all_trials.append(row_ranks)
            col_ranks_all_trials.append(col_ranks)

        return np.asarray(A_win_all_trials), np.asarray(SI_wins_all_trials), np.asarray(row_ranks_all_trials), np.asarray(col_ranks_all_trials)
    
    def sort_SI_wins(self, A_hat_all, SI_wins):
        SI_overall_all = []

        for A_hat in A_hat_all:
            # Computer A_mean over time wins
            A_mean = np.mean(A_hat, axis=2)
            SI_overall, _, _ = self._identifySS(A_mean)
            SI_overall_all.append(SI_overall)

        SI_overall_all = np.asarray(SI_overall_all)

        SI_sort_idx_overall = np.argsort(SI_overall_all.mean(0))
        SI_wins_sorted = SI_wins[SI_sort_idx_overall, :]
        labels_sort = [self._filtered_elec_areas[i] + " | " + self._filtered_elec_names[i] for i in SI_sort_idx_overall]

        return SI_wins_sorted, labels_sort
    
    def plot_SI_heatmap(self, A_hat_all, SI_wins, winSize, fs, time, out_path_plots=None):
        t_W = np.arange(0, self.nWin, 1)
        t_sec = np.arange(0, self.nWin*winSize/fs, 0.5)

        t_W_fig = t_W.copy()
        t_W_fig = t_W_fig[::30*2] - t_W_fig[0]

        SI_wins_sorted, labels_sort = self.sort_SI_wins(A_hat_all, SI_wins) 

        fig, axs = plt.subplots(1,1,figsize=(24, 15))
        sns.heatmap(SI_wins_sorted, ax=axs, xticklabels=time, yticklabels=labels_sort,cmap=sns.color_palette("rainbow", as_cmap=True), cbar_kws={"pad": 0.01, 'label': 'Sink Index'})
        sns.set(font_scale=0.8)

        axs.set_title('Sink index over time for all channels')
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Channel Area | Channel Name')

        plt.grid(False)
        plt.tight_layout()

        if out_path_plots is not None:
            plt.savefig(out_path_plots + '_SI_difference_heatmaps.png', bbox_inches = 'tight')
            plt.show()

    def SI_t_test(self, SI_wins_all, y):
        unique_ch_areas, ch_area_idxs = self._get_unique_channels()

        high_bet_trials = np.where(y == 1)[0]
        low_bet_trials = np.where(y == 0)[0]

        SI_wins_high_bet = SI_wins_all[high_bet_trials,:,:]
        SI_wins_low_bet = SI_wins_all[low_bet_trials,:,:]

        SI_wins_high_bet_all_areas = []
        SI_wins_low_bet_all_areas = []

        for idxs in ch_area_idxs:
            # Get the SI values for the high and low bet trials for the current area, average across time window, and flatten
            SI_wins_high_bet_area = SI_wins_high_bet.mean(axis=-1)
            SI_wins_low_bet_area = SI_wins_low_bet.mean(axis=-1)

            SI_wins_high_bet_area = SI_wins_high_bet_area[:,idxs[0]].flatten()
            SI_wins_low_bet_area = SI_wins_low_bet_area[:,idxs[0]].flatten()

            SI_wins_high_bet_all_areas.append(SI_wins_high_bet_area)
            SI_wins_low_bet_all_areas.append(SI_wins_low_bet_area)

        t_stats = []
        p_vals = []

        for i in range(len(unique_ch_areas)):
            t_stat, p_val = ttest_ind(SI_wins_high_bet_all_areas[i], SI_wins_low_bet_all_areas[i], equal_var=False)
            t_stats.append(t_stat)
            p_vals.append(p_val)

        return t_stats, p_vals, unique_ch_areas

    def plot_p_vals(self, SI_wins_all, y, out_path_plots=None):
        _, p_vals, unique_ch_areas = self.SI_t_test(SI_wins_all, y)

        fig, axs = plt.subplots(1, figsize=(15,10))

        axs.scatter(np.arange(len(unique_ch_areas)), p_vals)
        axs.set_xticks(np.arange(len(unique_ch_areas)), labels=unique_ch_areas,rotation=-90)
        axs.set_ylabel('P-Value')
        axs.set_xlabel('Channel Area')
        axs.set_title('P-Values for T-Test of SI Values for High and Low Bet Trials')

        if out_path_plots is not None:
            plt.savefig(out_path_plots + '_SI_p_vals.png', bbox_inches = 'tight')
            plt.show()

    def plot_t_stats(self, SI_wins_all, y, out_path_plots=None):
        t_stats, _, unique_ch_areas = self.SI_t_test(SI_wins_all, y)

        fig, axs = plt.subplots(1, figsize=(15,10))

        axs.scatter(np.arange(len(unique_ch_areas)), t_stats)
        axs.set_xticks(np.arange(len(unique_ch_areas)), labels=unique_ch_areas,rotation=-90)
        axs.set_ylabel('T-Stat')
        axs.set_xlabel('Channel Area')
        axs.set_title('T-Stat for T-Test of SI Values for High and Low Bet Trials')

        if out_path_plots is not None:
            plt.savefig(out_path_plots + '_SI_t_stats.png', bbox_inches = 'tight')
            plt.show()