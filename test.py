#%%
import numpy as np
from scipy.stats import linregress
from data_gen import load_MovieLens_1m_dataset, convert_to_cf_data, training_test_set
import tensorflow as tf
import os
import pickle

# Set hyperparameter and learning configuration
learning_rate = 10
epochs = 20000
global_step = 1000
model_name = "mf_model"
model_path = os.path.join(os.getcwd(), "model")
dataset_path = os.path.join(os.getcwd(), "dataset/mf_dataset.pkl")
training_session = True

features = 20

# Load dataset
if os.path.exists(dataset_path):    
    with open(dataset_path, "rb") as f:
        dataset, training_set, test_set, train_rated_vector, test_rated_vector = pickle.load(f)
else:
    features, target_values, users, movies, ratings, ml_data = load_MovieLens_1m_dataset()
    dataset, rated_vector = convert_to_cf_data(ml_data)
    training_set, test_set, train_rated_vector, test_rated_vector = training_test_set(dataset, rated_vector)
    
    # Save training and test data
    with open(dataset_path, "wb") as f:
        pickle.dump((dataset, training_set, test_set, train_rated_vector, test_rated_vector), f)


# indices = np.where(np.reshape(train_rated_vector, -1) == True)[0]
# test = np.reshape(dataset, -1)[indices].astype(int)

# P = tf.constant(dataset, dtype=tf.float32)
# P = tf.reshape(P, [-1])
# Q = tf.gather(P, indices)

tf.Variable(initial_value=np.array([1, 2]))
tf.Variable(initial_value=np.array())
init = tf.compat.v1.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    t = sess.run(Q)
    print(t)
print("A")