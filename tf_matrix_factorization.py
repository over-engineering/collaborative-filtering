from data_gen import load_MovieLens_1m_dataset_v2
import tensorflow as tf
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split
from utils import progressBar

# Set hyperparameter and learning configuration
parser = argparse.ArgumentParser(description="Learning parameters")
parser.add_argument("--learning_rate", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--dataset_path", type=str, default=os.path.join(os.getcwd(), "dataset/CF_dataset.pkl"))
parser.add_argument("--checkpoint", type=str, default=os.path.join(os.getcwd(), "model/matrix_factorization/tf"))
parser.add_argument("--model_name", type=str, default="mf_model")
parser.add_argument("--latent_factor", type=int, default=100) # 25 ~ 1000
parser.add_argument("--reg_lambda", type=float, default=0.002)
parser.add_argument("--cost_function", type=str, default="ALS")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--min_cost", type=float, default=2.0)

args = parser.parse_args()

class MF:
    def __init__(self, sess, shape, args):
        self.sess = sess
        num_user, num_item = shape
        reg_lambda = tf.constant(args.reg_lambda, dtype=tf.float32)
        learning_rate = args.learning_rate
        latent_factor = args.latent_factor
        self.cost_function = cost_function = args.cost_function
        self.checkpoint = checkpoint = args.checkpoint
        self.model_name = args.model_name
        self.epochs = args.epochs
        self.min_cost = args.min_cost

        # Model
        # R(u,i) = b(u) + b(i) + p(u) + p(q) + error

        self.H_indices = H_indices = tf.placeholder(tf.int32, shape=[None], name="user_ids")
        self.W_indices = W_indices = tf.placeholder(tf.int32, shape=[None], name="item_ids")

        self.H = H_embed = tf.Variable(tf.compat.v1.truncated_normal([num_user, latent_factor], stddev=0.2, mean=0), name="user_embedding") # User feature tensor
        self.W = W_embed = tf.Variable(tf.compat.v1.truncated_normal([num_item, latent_factor], stddev=0.2, mean=0), name="item_embedding") # Item feature tensor
        
        H_bias_embed = tf.Variable(tf.zeros([num_user]), dtype=tf.float32) # User bias
        W_bias_embed = tf.Variable(tf.zeros([num_item]), dtype=tf.float32) # Item bias
        
        H_vec = tf.nn.embedding_lookup(H_embed, H_indices)
        W_vec = tf.nn.embedding_lookup(W_embed, W_indices)

        H_bias_vec = tf.nn.embedding_lookup(H_bias_embed, H_indices)
        W_bias_vec = tf.nn.embedding_lookup(W_bias_embed, W_indices)

        R = tf.reduce_sum(tf.multiply(H_vec, W_vec), axis=1)
        R = tf.add(R, H_bias_vec)
        Y_ = tf.add(R, W_bias_vec)

        self.Y = Y = tf.placeholder(tf.float32, shape=[None], name="rating")

        # Cost function
        if cost_function is "ALS":
            # Good performance for explicit feedback dataset
            # ALS = MSE + l2_H+ l2_W + l2_H_bias + l2_W_bias
            l2_error = tf.multiply(reg_lambda, tf.add(tf.nn.l2_loss(H_embed), tf.nn.l2_loss(W_embed)))
            MSE = tf.losses.mean_squared_error(Y, Y_)
            self.cost = tf.add(MSE, l2_error)
        elif cost_function is "MAE":
            self.cost = tf.keras.losses.MAE(Y, Y_)
        elif cost_function is "MSE":
            self.cost = tf.losses.mean_squared_error(Y, Y_)

        # Optimizer
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)
        self.train = optimizer.minimize(self.cost)

        # Model save configuration
        self.saver = saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint)

        # Load or Save
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

    def fit(self, x, y, x_val, y_val, steps = 10000, save_best_only=True):
        sess, saver = self.sess, self.saver
        train, cost = self.train, self.cost
        H_indices, W_indices, Y = self.H_indices, self.W_indices, self.Y
        epochs, cost_function, min_cost = self.epochs, self.cost_function, self.min_cost
        checkpoint, model_name = self.checkpoint, self.model_name

        for epoch in range(0, epochs + 1):
            for step in range(0, steps + 1):
                training_error = sess.run(cost, feed_dict={H_indices: x[0], W_indices: x[1], Y: y})
                validation_error = sess.run(cost, feed_dict={H_indices: x_val[0], W_indices: x_val[1], Y: y_val})
                
                progressBar(step, steps, "Epoch : %d / %d, %s(training) : %.4f, %s(test) : %.4f" % (epoch, epochs, cost_function, training_error, cost_function, validation_error))
                
                sess.run(train, feed_dict={H_indices: x[0], W_indices: x[1], Y: y})
            
            if not save_best_only or (save_best_only and min_cost > validation_error):
                filename = model_name + "%03d_%.4f" % (epoch, validation_error) + ".ckpt"
                model_path = os.path.join(checkpoint, filename)
                saver.save(sess, model_path)
                print("Saved model %s" % filename)

        print("Done")

    def predict(self, x):
        pass

with tf.Session() as sess:
    num_user, num_item, train, test = load_MovieLens_1m_dataset_v2(target="cf")
    mf = MF(sess, shape=(num_user, num_item), args=args)
    train, validation = train_test_split(train, test_size=0.3)
    mf.fit(x=[train.userId, train.movieId], y=train.rating, x_val=[validation.userId, validation.movieId], y_val=validation.rating)