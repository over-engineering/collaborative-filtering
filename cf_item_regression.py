#%%
import numpy as np
from scipy.stats import linregress
from data_gen import load_MovieLens_1m_dataset, convert_to_cf_data, training_test_set
import tensorflow as tf
import os

#%%
features, target_values, users, movies, ratings, ml_data = load_MovieLens_1m_dataset()
dataset, rated_vector = convert_to_cf_data(ml_data)
num_user, num_item = dataset.shape

epochs = 20000

training_set, test_set, train_rated_vector, test_rated_vector = training_test_set(dataset, rated_vector)
mean_U = np.nanmean(training_set, axis = 0)
n_train = training_set - mean_U

#%%
X = tf.placeholder(tf.float32, shape=(num_user, num_item))
W = tf.Variable(tf.random.truncated_normal([num_item, num_item]), dtype=tf.float32) # item x item
b = tf.Variable(tf.random.truncated_normal([num_user, 1]), dtype=tf.float32) # user x 1
R = tf.matmul(X, W) + b
Y = tf.placeholder(tf.float32, shape=(test_set[test_rated_vector].shape))
Y_ = tf.placeholder(tf.float32, shape=(test_set[test_rated_vector].shape))

#%%
cost = tf.reduce_mean(tf.square(X - R))
learning_rate = 3
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)
# mae = tf.compat.v1.metrics.mean_absolute_error(Y, Y_)
mae = tf.keras.losses.MAE(Y, Y_)

#%%
global_step = 1000
model_path = os.path.join(os.getcwd(), "model")
model_name = "regression_model"
saver = tf.compat.v1.train.Saver()
ckpt = tf.train.get_checkpoint_state(model_path)

#%%
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    for epoch in range(epochs + 1):
        sess.run(train, feed_dict={X: n_train})
        if epoch % global_step == 0:
            loss = sess.run(cost, feed_dict={X: n_train})
            result = sess.run(R, feed_dict={X: test_set})
            error = sess.run(mae, feed_dict={Y: test_set[test_rated_vector], Y_: result[test_rated_vector]})
            print("Epoch : %d / %d, Loss : %f, MAE : %f" % (epoch, epochs, loss, error))

            saver.save(sess, os.path.join(model_path, model_name + ".ckpt"), global_step = global_step)
