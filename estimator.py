class Estimator(object):
    def __init__(self, data, setup_data):
        self._num_trials, self._num_channels, self._num_freqs, self._num_timesteps = data.shape
        
        self._elec_areas = setup_data['elec_area']
        self._elec_names = setup_data['elec_name']

    def create_X(self, data, channel, time):
        """Create the X data that will be used to train the LDA model"""
        if hasattr(self, '_time_resolution') and type(self._time_resolution) == int:
            X = data[:, channel, :, time:time+self._time_resolution].mean(-1)
        elif type(time) == list and len(time) == 2:
            if time[0] - time[1] == 0:
                X = data[:, channel, :, time[0]]
            else:
                X = data[:, channel, :, time[0]:time[1]].mean(-1)
        else:
            pass

        return X
    
    def filter_channels(self):
        """Filters out channels that are in particular anatomical locations"""
        filtered_elec_areas_idxs = [i for i,ea in enumerate(self._elec_areas) if ea not in 
                                    ['white matter','CZ','PZ', 'out','FZ','cerebrospinal fluid',
                                     'lesion L','ventricle L','ventricle R']]
        filtered_elec_areas = [self._elec_areas[i] for i in filtered_elec_areas_idxs]
        filtered_num_channels = len(filtered_elec_areas_idxs)

        return filtered_elec_areas_idxs, filtered_elec_areas, filtered_num_channels
    
    def set_attributes(self, **kwargs):
        """Set class attributes specified by the dataset and metadata"""
        if 'time_resolution' in kwargs:
            if not(self._num_timesteps % kwargs['time_resolution'] == 0):
                raise Exception("Invalid time resolution size, num_timesteps % resolution must equal 0")
            else:
                self._time_resolution = kwargs['time_resolution']
                self._timesteps_rescaled = int(self._num_timesteps/kwargs['time_resolution'])