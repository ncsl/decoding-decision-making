from itertools import combinations
import numpy as np
from LDA import LDA

def _find_combinations(n, k):
    # Find all the different combinations of k channels
    population = list(range(0, n))
    combinations_list = list(combinations(population, k))
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


class GridSearch(object):
    """Class for performing grid search on the optimal channel combination to use for collective prediction
    
    LDA must be trained using the method train_on_optimal_time_windows() before using this class
    """

    def __init__(self, lda:LDA):
        self.__lda = lda

    def _get_predictions(self):
        predictions = []

        # Get predictions for each trial by each channel
        for trial in range(self.__lda.num_trials):
            trial_predictions = []
            for dval in self.__lda.dvals[:,trial]:
                if dval >= 0:
                    channel_prediction = 1
                else:
                    channel_prediction = 0
                trial_predictions.append(channel_prediction)
            
            predictions.append(trial_predictions)

        predictions = np.asarray(predictions)

        return predictions
    
    def _grid_search_on_channel_combinations(self, y):
        predictions = self._get_predictions()
            # Find all possible channel combinations to use for collective prediction
        all_channel_idxs_combinations = []
        for i in range(predictions.shape[1]):
            # Find all possible channel combinations of length i+1
            channel_combinations = _find_combinations(predictions.shape[1], i+1)
            all_channel_idxs_combinations.append(channel_combinations)

        # Grid search on optimal channel combination to use for collective prediction
        accuracies = []

        for k_length_combinations in all_channel_idxs_combinations:
            k_length_combination_accuracies = []
            for combination in k_length_combinations:
                # Stores all the predictions for this particular combination of k channels
                combination = list(combination)
                combination_predictions = predictions[:,combination]

                # Stores the collective prediction for this particular combination of k channels
                collective_combination_predictions = _get_collective_predictions(combination_predictions)

                # Get the accuracy of the collective prediction for each combination of k channels
                k_length_combination_accuracies.append(_get_collective_prediction_accuracy(collective_combination_predictions, y))
            
            accuracies.append(k_length_combination_accuracies)

        return all_channel_idxs_combinations, accuracies

    def get_optimal_channel_combination(self, y):
        all_channel_idxs_combinations, accuracies = self._grid_search_on_channel_combinations(y)

        optimal_time_windows_per_ch = self.__lda.optimal_time_windows_per_channel
        max_accuracies = []

        for i, accuracies in enumerate(accuracies):
            optimal_chs = []
            ch_idxs = all_channel_idxs_combinations[i][np.argmax(accuracies)]
            for idx in ch_idxs:
                optimal_chs.append(optimal_time_windows_per_ch[idx][0])

            max_accuracies.append([optimal_chs, np.max(accuracies)])

        max_accuracies.sort(key=lambda x: x[1], reverse=True)

        return max_accuracies