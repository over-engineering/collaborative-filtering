import numpy as np
from scipy.stats import linregress
from data_gen import load_MovieLens_1m_dataset, convert_to_cf_data, training_test_set
import tensorflow as tf
import os
import pickle
import argparse

# Set hyperparameter and learning configuration
parser = argparse.ArgumentParser(description="Learning parameters")
parser.add_argument("--learning_rate", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--global_step", type=int, default=1000)
parser.add_argument("--dataset_path", type=str, default=os.path.join(os.getcwd(), "dataset/mf_dataset.pkl"))
parser.add_argument("--checkpoint", type=str, default=os.path.join(os.getcwd(), "model/matrix_factorization"))
parser.add_argument("--model_name", type=str, default="mf_model")
parser.add_argument("--latent_factor", type=int, default=100) # 25 ~ 1000
parser.add_argument("--reg_lambda", type=float, default=0.002)

args = parser.parse_args()

class MF:
    def __init__(self, sess, shape, args):
        self.sess = sess
        num_user, num_item = shape
        reg_lambda = tf.constant(args.reg_lambda, dtype=tf.float32)
        learning_rate = args.learning_rate
        latent_factor = args.latent_factor
        self.checkpoint = checkpoint = args.checkpoint
        self.model_name = args.model_name
        self.global_step = args.global_step
        self.epochs = args.epochs

        self.indices = indices = tf.placeholder(tf.int32, [None]) # Lookup index list for sparse dataset
        self.Y = Y = tf.placeholder(tf.float32, [None])

        # Model
        H = tf.Variable(tf.compat.v1.truncated_normal([num_user, latent_factor], stddev=0.2, mean=0), name="user_feature_matrix") # User feature tensor
        W = tf.Variable(tf.compat.v1.truncated_normal([latent_factor, num_item], stddev=0.2, mean=0), name="item_feature_matrix") # Item feature tensor

        # R(u,i) = b(u) + b(i) + p(u) + p(q) + error
        H_bias = tf.Variable(np.zeros([num_user, 1]), dtype=tf.float32) # User bias
        W_bias = tf.Variable(np.zeros([1, num_item]), dtype=tf.float32) # Item bias

        R = tf.matmul(H, W)
        R = tf.add(R, H_bias)
        self.R = R = tf.add(R, W_bias)

        # Output   
        Y_ = tf.nn.embedding_lookup(tf.reshape(R, [-1]), indices)

        self.H = tf.add(H, H_bias)
        self.W = tf.add(W, W_bias)

        # Cost function and Optimizer

        # J = MSE + l2_H+ l2_W + l2_H_bias + l2_W_bias
        l2_error = tf.multiply(reg_lambda, tf.add(tf.nn.l2_loss(H), tf.nn.l2_loss(W)))
        mean_square_error = tf.reduce_mean(tf.square(tf.subtract(Y, Y_)))
        self.cost = cost = tf.add(mean_square_error, l2_error)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)

        # self.train = optimizer.minimize(cost)
        self.MAE = tf.keras.losses.MAE(Y, Y_)
        self.train = optimizer.minimize(self.MAE)

        # Model save configuration
        self.saver = saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint)


        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
    
    def fit(self, training_set, test_set, training_indices, test_indices):
        sess, saver =  self.sess, self.saver
        H, W, train = self.H, self.W, self.train
        MAE, cost = self.MAE, self.cost
        Y, indices = self.Y, self.indices # placeholders
        epochs, global_step, checkpoint, model_name = self.epochs, self.global_step, self.checkpoint, self.model_name

        for epoch in range(0, epochs + 1):
            if epoch % global_step == 0:
                # loss = sess.run(cost, feed_dict={Y: training_set, indices: training_indices})
                training_error = sess.run(MAE, feed_dict={Y: training_set, indices: training_indices})
                test_error = sess.run(MAE, feed_dict={Y: test_set, indices: test_indices})
                # print("Epoch : %d / %d, Loss : %f, MAE(training) : %f, MAE(test) : %f" % (epoch, epochs, loss, training_error, test_error))
                print("Epoch : %d / %d, MAE(training) : %f, MAE(test) : %f" % (epoch, epochs, training_error, test_error))
                saver.save(sess, os.path.join(checkpoint, model_name + ".ckpt"), global_step = global_step)
            
            sess.run(train, feed_dict={Y: training_set, indices: training_indices})
        
    def prediction(self, target):
        prediction = sess.run(self.R)
        print(np.reshape(prediction, [-1])[target])

    def test(self, dataset, indices):
        print(self.sess.run(self.Y, feed_dict={self.Y: dataset, self.indices: indices}))

def load_data():
    dataset_path = args.dataset_path
    # Load dataset
    if os.path.exists(dataset_path):    
        with open(dataset_path, "rb") as f:
            return pickle.load(f)
    else:
        dataset, rated_vector = load_MovieLens_1m_dataset(target="cf")
        (num_user, num_item, training_set, test_set, training_indices, test_indices) = training_test_set(dataset, rated_vector)
        mf_dataset = (num_user, num_item, training_set, test_set, training_indices, test_indices, dataset)
        try:
            return mf_dataset
        finally:
            # Save training and test data
            with open(dataset_path, "wb") as f:
                pickle.dump(mf_dataset, f)

with tf.Session() as sess:
    (num_user, num_item, training_set, test_set, training_indices, test_indices, dataset) = load_data()

    target = np.concatenate((training_indices, test_indices))
    args.epochs = 10000
    args.learning_rate = 10
    args.reg_lambda = 0.005

    mf = MF(sess, shape=(num_user, num_item), args=args)
    mf.fit(training_set, test_set, training_indices, test_indices)
    mf.prediction(target)
    print(np.reshape(dataset, [-1])[target])