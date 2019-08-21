import numpy as np
from math import sqrt, isnan
from multiprocessing import Pool

def pearson_correlation(i, j, exclude = False):
    '''
    Caluate similarity based on pearson correlaction

    :param i: Set of rating value of item i of all users
    :param j: Set of rating value of item j of all users
    :param mean_U: Set of mean rating value of all users to reflect user rating trend
    :param exclude: Exclude not evaluated item from items i and j

    :returns: The similarity value between items i and j
    '''
    if exclude:
        target = np.nonzero(np.logical_and(i, j))[0]
        i = i[target]
        j = j[target]

    numerator = np.sum((i - np.mean(i)) * (j - np.mean(j)))
    denominator = sqrt(np.sum(np.power(i - np.mean(i), 2)))\
        * sqrt(np.sum(np.power(j - np.mean(j), 2)))
    
    if denominator == 0 or isnan(denominator):
        return 0

    return numerator / denominator


def adjusted_cosine_similarity(i, j, mean_U, exclude=False):
    '''
    Caluate similarity based on 'Item-Based Collaborative Filtering Recommendation Algorithms' at WWW10

    :param i: Set of rating value of item i of all users
    :param j: Set of rating value of item j of all users
    :param mean_U: Set of mean rating value of all users to reflect user rating trend
    :param exclude: Exclude not evaluated item from items i and j

    :returns: The similarity value between items i and j
    '''

    if exclude:
        target = np.nonzero(np.logical_and(i, j))[0]
        i = i[target]
        j = j[target]
        mean_U = mean_U[target]

    numerator = np.sum((i - mean_U) * (j - mean_U))
    denominator = sqrt(np.sum(np.power(i - mean_U, 2)))\
        * sqrt(np.sum(np.power(j - mean_U, 2)))
    
    if denominator == 0 or isnan(denominator):
        return 0

    return numerator / denominator

def worker_user(output, data, i, j):
    output[i][j] = pearson_correlation(data[i, :], data[j, :], True)

def worker_item(output, data, i, j, mean_U):
    output[i][j] = adjusted_cosine_similarity(data[:, i], data[:, j], mean_U, True)

def get_similarity_matrix(data, target="item", numbuer_of_process = 20):
    pool = Pool(numbuer_of_process)

    number_of_users = data.shape[0]
    number_of_items = data.shape[1]
    if target == "user":
        output = np.zeros((number_of_users, number_of_users))
        return np.array([pearson_correlation(data[i, :], data[j, :], True) for j in range(0, number_of_users) for i in range(0, number_of_users)]).reshape(number_of_users, number_of_users)
    elif target == "item":
        mean_U = np.mean(data, axis=1)
        return np.array([adjusted_cosine_similarity(data[:, i], data[:, j], mean_U, True) for j in range(0, number_of_items) for i in range(0, number_of_items)]).reshape(number_of_items, number_of_items)
    return -1