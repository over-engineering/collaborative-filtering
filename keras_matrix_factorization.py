#%% Package import
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout, Dot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import MAE

from data_gen import load_MovieLens_1m_dataset_v2
import os

from sklearn.model_selection import train_test_split

import argparse

#%% Set hyperparameter and learning configuration
parser = argparse.ArgumentParser(description="Learning parameters")
parser.add_argument("--checkpoint", type=str, default=os.path.join(os.getcwd(), "model/matrix_factorization/keras"))
parser.add_argument("--model_name", type=str, default="mf_model")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--latent_factor", type=int, default=100) # 25 ~ 1000
parser.add_argument("--validation_split", type=float, default=0.2)
parser.add_argument("--save_best_only", type=bool, default=True)
parser.add_argument("--patience", type=int, default=30)
args = parser.parse_args()

#%% Model definition
class MF:
    def __init__(self, shape, args):
        # learning configuration
        self.epochs = args.epochs
        self.verbose = verbose = args.verbose
        self.validation_split = args.validation_split
        latent_factor = args.latent_factor
        num_user, num_item = shape

        model_path = os.path.join(args.checkpoint, args.model_name) + "{epoch:03d}-{val_loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss",
                                        verbose=verbose, save_best_only=args.save_best_only)
        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=verbose)
        self.callbacks = [checkpoint, early_stopping]

        H_input = Input((1,), dtype="int32")
        H_embed = Embedding(num_user, latent_factor, name="user")(H_input)
        H_vec = Flatten(name="user_flatten")(H_embed)

        W_input = Input((1,), dtype="int32")
        W_embed = Embedding(num_item, latent_factor, name="item")(W_input)
        W_vec = Flatten(name="item_flatten")(W_embed)

        R = Dot(axes=1)([H_vec, W_vec])

        self.model = Model(inputs=[H_input, W_input], outputs=R)
        self.model.compile("adam", metrics=["mae"], loss="mae")

    def fit(self, X_train, Y_train):
        history = self.model.fit(x=X_train, y=Y_train,
                                epochs=self.epochs, verbose=self.verbose,
                                validation_split=self.validation_split, callbacks=self.callbacks)
        return history
    
    def predict(self, X):
        return self.model.predict(x=X, verbose=self.verbose)

    def error(self, X):
        return MAE(X, self.predict(X))
        
    def summary(self):
        self.model.summary()

#%% Data preprocessing
num_user, num_item, train, test = load_MovieLens_1m_dataset_v2(target="cf")

#%%
mf = MF((num_user, num_item), args)
mf.fit([train.userId, train.movieId], train.rating)
prediction = mf.predict([test.userId, test.movieId])

#%%
