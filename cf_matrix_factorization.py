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
model_path = os.path.join(os.getcwd(), "model/matrix_factorization")
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

# Data preprocessing
num_user, num_item = dataset.shape
train_indices = np.where(np.reshape(train_rated_vector, -1) == True)[0]
test_indices = np.where(np.reshape(test_rated_vector, -1) == True)[0]
training_set = np.reshape(training_set, [-1])[train_indices]
test_set = np.reshape(test_set, [-1])[test_indices]

# Model
H = tf.Variable(tf.compat.v1.truncated_normal([num_user, features], stddev=0.2, mean=0), name="user_feature_matrix")
W = tf.Variable(tf.compat.v1.truncated_normal([features, num_item], stddev=0.2, mean=0), name="item_feature_matrix")

# R(u,i) = b(u) + b(i) + p(u) + p(q) + error
H_bias = tf.Variable(np.zeros([num_user, 1]), dtype=tf.float32)
W_bias = tf.Variable(np.zeros([1, num_item]), dtype=tf.float32)

R = tf.matmul(H, W)
R = tf.add(R, H_bias)
R = tf.add(R, W_bias)

# Output
Y_hat_train = tf.gather(tf.reshape(R, [-1]), train_indices)
Y_train = tf.placeholder(tf.float32, train_indices.shape)
Y_hat_test = tf.gather(tf.reshape(R, [-1]), test_indices)
Y_test = tf.placeholder(tf.float32, test_indices.shape)

# Cost function and Optimizer
cost = tf.reduce_mean(tf.square(tf.subtract(Y_train, Y_hat_train)))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)
training_mae = tf.keras.losses.MAE(Y_train, Y_hat_train)
test_mae = tf.keras.losses.MAE(Y_test, Y_hat_test)

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
                loss = sess.run(cost, feed_dict={Y_train: training_set})
                training_error, test_error = sess.run([training_mae, test_mae], feed_dict={Y_train: training_set, Y_test: test_set})
                print("Epoch : %d / %d, Loss : %f, MAE(training) : %f, MAE(test) : %f" % (epoch, epochs, loss, training_error, test_error))
                saver.save(sess, os.path.join(model_path, model_name + ".ckpt"), global_step = global_step)
            
            sess.run(train, feed_dict={Y_train: training_set})
    else:
        prediction = sess.run(R)
        print(prediction)