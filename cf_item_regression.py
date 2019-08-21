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
model_name = "regression_model"
model_path = os.path.join(os.getcwd(), "model")
dataset_path = os.path.join(os.getcwd(), "dataset/regression_dataset.pkl")
training_session = True

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

# Data preprocessing
num_user, num_item = dataset.shape
mean_U = np.nanmean(training_set, axis = 0)
n_train = training_set - mean_U

# Model
X = tf.placeholder(tf.float32, shape=(num_user, num_item))
W = tf.Variable(tf.random.truncated_normal([num_item, num_item]), dtype=tf.float32) # item x item
b = tf.Variable(tf.random.truncated_normal([num_user, 1]), dtype=tf.float32) # user x 1
R = tf.matmul(X, W) + b

# Prediction and True value placeholder
Y = tf.placeholder(tf.float32, shape=(test_set[test_rated_vector].shape))
Y_ = tf.placeholder(tf.float32, shape=(test_set[test_rated_vector].shape))

# Cost function and Optimizer
cost = tf.reduce_mean(tf.square(X - R))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)
mae = tf.keras.losses.MAE(Y, Y_)

# Model save configuration
saver = tf.compat.v1.train.Saver()
ckpt = tf.train.get_checkpoint_state(model_path)

# Learn model
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    if training_session:
        for epoch in range(0, epochs + 1):
            if epoch % global_step == 0:
                loss = sess.run(cost, feed_dict={X: n_train})
                result = sess.run(R, feed_dict={X: test_set})
                error = sess.run(mae, feed_dict={Y: test_set[test_rated_vector], Y_: (result + mean_U)[test_rated_vector]})
                
                print("Epoch : %d / %d, Loss : %f, MAE : %f" % (epoch, epochs, loss, error))

                saver.save(sess, os.path.join(model_path, model_name + ".ckpt"), global_step = global_step)

            sess.run(train, feed_dict={X: n_train})
    else:
        prediction = sess.run(R, feed_dict={X: dataset})
        print(prediction)