import numpy as np
from math import isnan
import similarity

import tensorflow as tf

def weighted_sum(data, sim_matrix, axis = 0):
    '''
    Compute the prediction on an item i for a user u

    :param i: Set of user or item rating values
    :param sim: Set of similarity values
    :param axis: Sum direction

    :returns: ratings predicted using weighted sum
    '''

    number_of_users = data.shape[0]
    number_of_items = data.shape[1]

    output = data

    # Select the position not evaluated
    targets = np.argwhere(data == 0)

    for t in targets:
        user = t[0]
        item = t[1]

        if axis == 0: # item-based
            target = sim_matrix[item, :] != 1
            output[user][item] = np.sum(sim_matrix[item, :][target] * data[user, :][target]) / np.sum(np.abs(sim_matrix[item, :][target]))
        elif axis == 1: # user-based
            target = sim_matrix[user, :] != 1
            output[user][item] = np.sum(sim_matrix[user, :][target] * data[:, item][target]) / np.sum(np.abs(sim_matrix[user, :][target]))
        else:
            raise "Invalid Value"

    return output

def prediction(input_data, target="item", method="weighted_sum"):
    '''
    Predict value not rated

    :param input_data: rating data to be predicted
    :param method: prediction method(weighted_sum, linear_regression, ...)

    :returns: predicted rating value
    '''

    sim = similarity.get_similarity_matrix(input_data, target=target)

    axis = 0 if target == "item" else 1
        
    if method == "weighted_sum":
        return weighted_sum(input_data, sim, axis=axis)
    elif method == "linear_regression":
        pass
