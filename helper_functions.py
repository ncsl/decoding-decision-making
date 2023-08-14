import numpy as np
import itertools

def _generate_sampled_channels(channels, sample_size, sampled_channels_=[]):
    """Recursively generate a list (size = sample_size) of lists of sampled channels"""
    np.random.seed()

    if len(channels) > sample_size:
        sample = np.random.choice(channels, size=sample_size, replace=False)
        sampled_channels_.append(list(sample))
        channels = np.delete(channels, np.where(np.isin(channels, sample))[0])
        _generate_sampled_channels(channels, sample_size, sampled_channels_)
    else:
        sampled_channels_.append(list(channels))

    return sampled_channels_

def _find_combinations(n, k):
    """Find all possible combinations of k channels from n channels

    Return
    ------
    combinations_list : list
        List of tuples containing the indices all possible combinations of k channels from n channels
    """
    population = list(range(0, n))
    combinations_list = list(itertools.combinations(population, k))
    return combinations_list

def _get_collective_predictions(predictions):
    # Get the collective prediction for each trial
    collective_predictions = []

    for trial_predictions in predictions:
        if trial_predictions.mean() >= 0.5:
            collective_predictions.append(1)
        else:
            collective_predictions.append(0)
    
    return collective_predictions

def _get_collective_prediction_accuracy(collective_predictions, y):
    # Get the accuracy of theveloped a circuit model of decision-making which accounts for the specificity of inputs to and outputs from inhibitory neurons. We found that selective inhibition expands the space of circuits supporting decision-making, allowing for weaker or stronger recurrent excitation when connected in a competitive or feedback motif. The specificity of inhibitory outputs sets te collective prediction
    accuracy = (y == collective_predictions).mean()
    return accuracy