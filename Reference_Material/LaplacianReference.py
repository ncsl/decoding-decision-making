#%%
import h5py 
import mat73
import numpy as np
from matplotlib import pyplot as plt
import re
import mne
from scipy import signal,stats


#%%
#setup_path = '/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT06/EFRI06_WAR_SES1_Setup.mat'
#raw_path = '/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT06/EFRI06_WAR_SES1_Raw.mat'

# subs = ['06','07','10','12','13','15','16','17','18','21']
# subs = ['17','18','21']
subs=['16']
for sub in subs:
    setup_path = f'/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Setup.mat'
    raw_path = f'/mnt/ncsl_share/Public/EFRI/1_formatted/SUBJECT{sub}/EFRI{sub}_WAR_SES1_Raw.mat'
    out_path = '/mnt/ncsl_share/Daniel/Data/'
    setup_data = mat73.loadmat(setup_path)
    raw_file = h5py.File(raw_path)
    Fs = raw_file['Fs'][0][0]
    lfp_data = raw_file['lfpdata']
    lfp_all = lfp_data[:,:]
    elec_names = setup_data['elec_name']
    # Filter Data Bandpass .5-200 Hz 
    filt = mne.filter.filter_data(lfp_all,Fs,0.5,200,method="iir")
    # notch filter 60 hz harmonics
    for notchfreq in [60,120,180]:
        filt = mne.filter.notch_filter(filt,Fs,notchfreq, method="iir")
    # decimate to 500 Hz 
    decFactor = int(Fs/500)
    filt = filt[:,::decFactor]

    ## For each channel in elec_names, get its index position in array, whether its on the end of the electrode shaft, and its neighboring indices 
    lap_ref_data = np.zeros(filt.shape)
    for i,en in enumerate(elec_names):
        if en in ["REF1","REF2","E","CZ","FZ","PZ"]:
            lap_ref_data[i,:] = filt[i,:]
            continue
        pattern = r"([a-z']+) *(\d+)"
        shaft_name = re.findall(pattern,en,re.IGNORECASE)[0][0]
        elec_num = re.findall(pattern,en,re.IGNORECASE)[0][1]
        en_plus1 = f"{shaft_name}{str(int(elec_num)+1)}"
        en_minus1 = f"{shaft_name}{str(int(elec_num)-1)}"
        if en_minus1 not in elec_names:
            neighbor_inds=[i-1]
        elif en_plus1 not in elec_names:
            neighbor_inds=[i+1]
        else:
            neighbor_inds = [ i-1,i+1]
        print(en, i, neighbor_inds,[elec_names[n] for n in neighbor_inds])
        neighbor_mean = np.mean(filt[neighbor_inds,:],axis=0)
        lap_ref_data[i,:] = filt[i,:] - neighbor_mean
    
    # Wavelet 
    wavlet_freqs = np.logspace(np.log2(2),np.log2(150),num=63,base=2)
    #tfr_by_chan_list = []    
    tfr_array = np.zeros((lap_ref_data.shape[0], len(wavlet_freqs), int(lap_ref_data.shape[1]/25)))
    for ch in range(lap_ref_data.shape[0]):
        tfr = mne.time_frequency.tfr_array_morlet(lap_ref_data[ch,:].reshape(1,1,-1),500,wavlet_freqs,n_cycles=6,output='power',n_jobs=1,)
        #
        #downsample = np.convolve(tfr[0,0,0,:], np.ones(50, ), mode='valid')[::25]/50. #Fs = 500; 1/Fs = .002 s; .1 sec/.002 sec = 50 samples for 100 ms. hence 50. then downsample every 25 because going from 500 Hz to 20 Hz final time resolution
        downsample = signal.convolve2d(tfr[0,0,:,:], (1.0/50)* np.ones((1,50)), mode='same')[:,::25]
        #tfr_by_chan_list.append(downsample)
        downsample = np.log(downsample) # natural log normalization to make frequencies more comparable
        downsample = stats.zscore(downsample,axis=1)
        tfr_array[ch,:,:] = downsample
        print(ch)
        #if ch>=2:
        #    break 

    ## Snapshot around start move 
    dsFs = 20 #downsample FS is 20 Hz 
    good_trials = setup_data['filters']['trial'][setup_data['filters']['success']].astype(int)-1
    num_trials = len(good_trials)
    trials_by_channels_by_freqs_by_time_array = np.zeros((num_trials,tfr_array.shape[0],tfr_array.shape[1],int(5*dsFs)))
    for i,t in enumerate(good_trials):
        start_move_time = setup_data['trial_times'][t][0][setup_data['trial_words'][t][0]==35][0]
        #print(f'start move time = {start_move_time} for trial {t}')
        
        ## To go from the time to the index position in the lfp array, multiply time by Fs 
        start_move_index = int(start_move_time*dsFs)
        #print(f'start move index = {start_move_index} for trial {t}')
        start_index = int((start_move_time - 3.0)*dsFs)
        end_index = start_index+int(5*dsFs)#int((start_move_time + 2.0)*Fs)
        data_slice = tfr_array[:,:,start_index:end_index]
        trials_by_channels_by_freqs_by_time_array[i,:,:,:] = data_slice
        #break

    np.save(f'{out_path}Trial_by_Chan_by_Freq_by_Time_Snapshots/Subject{sub}_snapshot_normalized.npy',trials_by_channels_by_freqs_by_time_array)
    #break
#%% 