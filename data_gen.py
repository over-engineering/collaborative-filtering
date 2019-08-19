"""
=====================================
:mod: `data generator module`
=====================================

description
=====================================
Generate random dataset to train and validate recommandation algorithm
"""

from urllib import request
import os
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle # Module to save non-text structured data
import utils

path = os.path.join(os.getcwd(), "dataset")
movielens_path = os.path.join(path, "movielens")

user_name = "userId"
movie_name = "movieId"
rating_name = "rating"

try:
    os.makedirs(movielens_path)
except OSError:
    print("MobieLens directory exists")

def convert_to_cf_data(data):
    '''
    Convert raw merged data to user x item rating Matrix

    :param data: MovieLens merged data

    :returns: converted data for collaborative filtering 
    '''
    
    converted_data = data.pivot(index=user_name, columns=movie_name, values=rating_name)
    converted_data = converted_data.fillna(0) # Nan -> 0
    return converted_data
def load_MovieLens_1m_dataset():
    '''
    Load MovieLens 1m dataset

    :returns: features, target_values, users, movies, ratings, data
    '''
    
    dataset = "ml-1m"
    data_path = os.path.join(movielens_path, dataset)
    pkl_path = os.path.join(data_path, "%s.pkl" % dataset)

    if not os.path.exists(pkl_path):
        url = "http://files.grouplens.org/datasets/movielens/%s.zip" % dataset

        print("Downloading MovieLens %s dataset..." % dataset)
        utils.download(url, data_path + ".zip")
        utils.unzip("%s.zip" % data_path, movielens_path)
        
        # Read data
        user_names = [user_name, "gender", "age", "occupationId", "zip-code"]
        movie_names =  [movie_name, "Title", "Genres"]
        rating_names = [user_name, movie_name, rating_name, "timestamp"]

        users = pd.read_csv(os.path.join(data_path, "users.dat"), sep="::", header=None, names=user_names, engine="python")
        movies = pd.read_csv(os.path.join(data_path, "movies.dat"), sep="::", header=None, names=movie_names, engine="python")
        ratings = pd.read_csv(os.path.join(data_path, "ratings.dat"), sep="::", header=None, names=rating_names, engine="python")
        
        # Preprocessing

        # Remove unnecessary data
        ratings = ratings.filter(regex='userId|movieId|rating')

        # Merge data
        data = pd.merge(pd.merge(ratings, users), movies)

        # Separate features and rating values
        features = data.drop([rating_name], axis = 1).values # drop rating column
        target_values = data[rating_name].values

        preprocessed_data = (features, target_values, users, movies, ratings, data)

        # Save preprocessed data
        with open(pkl_path, "wb") as f:
            pickle.dump(preprocessed_data, f)
        
        return preprocessed_data
    
    else:
        with open(pkl_path, mode="rb") as f:
            return pickle.load(f)

# Test
if __name__ == "__main__":
    features, target_values, users, movies, ratings, data = load_MovieLens_1m_dataset()
    cf_data = convert_to_cf_data(data)
    print(cf_data.shape)
