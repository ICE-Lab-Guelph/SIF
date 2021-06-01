# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Tuning A Neural Network Using SIF vs KF (Classification Task)

# %%
# Installing necessary libraries

# Importing global modules
from pprint import pformat
from sklearn import datasets
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from filterpy.kalman import (
    KalmanFilter,
    UnscentedKalmanFilter,
    MerweScaledSigmaPoints,
    unscented_transform,
)
from keras.models import Sequential
from keras.layers import Dense, Dropout
import math
import os
import time
import logging
from sklearn.metrics import accuracy_score
from keras.callbacks import Callback

# import matlab.engine
from io import StringIO
import pdb
import tensorflow as tf
import random

# eng = matlab.engine.start_matlab()

# Importing local modules
import ukf
import utility


# %%
# Tracking of weight records of every epochs
class EpochInfoTracker(Callback):
    def __init__(self):
        self.weights_history = []  # Tracking the weights in each epochs

    def on_epoch_end(self, epoch, logs=None):
        weights_vec = get_weights_vector(self.model)
        self.weights_history.append(weights_vec)


# Class for storing the necessary parameters
class Params:
    pass


# %% [markdown]
# ## Loading Iris Dataset

# %%
iris = datasets.load_iris()  # Load iris dataset

# Create X and y of dataframe
X = iris.data[:, :4]  # X dataset
y = np.asarray(pd.get_dummies(iris.target))  # y dataset

print("X dataset shape:", X.shape)
print("y dataset shape:", y.shape)


# %%
from sklearn.model_selection import train_test_split

# Prepare training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# %% [markdown]
# ## Initialize Essential Functions and Parameters for the Algorithm

# %%
# --------------------Initialization of the parameters-----------------
params = Params()
params.epochs = 1200
params.train_series_length = X_train.shape[0]
params.test_series_length = X_test.shape[0]
params.mg_tau = 30
# params.window_size = 12    # M
params.ukf_dt = 0.1
params.alpha, params.beta, params.kappa = 1, 2, 1  # Worked well
# params.alpha, params.beta, params.kappa = 0.001, 2, 1
params.Q_var = 0.001
params.R_var = 0.001

# To make training data and related variables accessible across functions
params.train_ukf_ann = True
params.X_data = None
params.y_data = None
params.hxw_model = None
params.curr_idx = 0


# ---------------- Initialization of the necessary functions------------------
def measurement_func(w, x):
    hxw_model = params.hxw_model
    qq = np.asarray(w)
    ww = np.reshape(qq, -1)
    set_weights(hxw_model, ww)
    # Reshape needed to feed x as 1 sample to ANN model
    hxw = hxw_model.predict(x.reshape(1, len(x)))
    hxw = hxw.flatten()  # Flatten to make shape = (1,)


# Create ukf using pykalman library
def create_ukf(Q, R, dt, w_init, P_init):
    M = w_init.shape[0]

    points = MerweScaledSigmaPoints(M, params.alpha, params.beta, params.kappa)

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=1, dt=dt, fx=fw, hx=hw, points=points)
    ukf.x = w_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


# Create ukf instance using ukf.py (custom ukf)
def create_my_ukf(Q, R, dt, w_init, P_init):
    my_ukf = ukf.UnscentedKalmanFilter(
        fw, hw, R, Q, w_init, P_init, params.alpha, params.beta, params.kappa
    )
    return my_ukf


# Function for Kalman filter
def fw(w, dt=None):
    return w  # Identity


# Function for Kalman filter
def hw(w):
    x = params.X_data[params.curr_idx]
    hxw = measurement_func(w, x)
    return hxw


def evaluate_neural_nets(
    sgd_ann, ukf_ann, window, use_train_series=False, train_series=None
):
    if use_train_series:
        X_data, y_data = X_train, y_train
        series = train_series
        sample_len = params.train_series_length
        title = "Train series (true vs. predicted)"
    else:
        sample_len = params.test_series_length
        X_data, y_data = X_test, y_test
        title = "Test series (true vs. predicted)"

    sgd_pred, sgd_self_pred = utility.predict_series(
        sgd_ann, X_data, sample_len, window
    )
    ukf_pred, ukf_self_pred = utility.predict_series(
        ukf_ann, X_data, sample_len, window
    )

    utility.plot(range(sample_len), series, title=title, label="True series")
    # utility.plot(range(sample_len), sgd_pred, new_figure=False, label='SGD ANN prediction (based on true windows)')
    utility.plot(
        range(sample_len), ukf_pred, new_figure=False, label="SIF ANN prediction"
    )
    if not use_train_series:
        preds = ukf_ann.predict(X_data)
        accuracy = accuracy_score(y_data, preds)
        print("The Test accuracy is: ", accuracy)

    # utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
    #              label='Predicted test series (rolling prediction: no true vals used)')


# Create a simple feedforward neural network
def create_neural_net(M):

    # Build a simple neural network
    ann = Sequential()
    ann.add(Dense(1, input_dim=M, activation="relu"))
    ann.add(Dense(3, activation="softmax"))
    ann.compile(optimizer="sgd", loss="categorical_crossentropy", metrics="accuracy")

    # Print out the summary of the model
    ann.summary()

    return ann


# Get weights of the neural network model
def get_weights_vector(model):
    weights = model.get_weights()
    # logging.info(weights)
    weights_vec = []
    for w_mat in weights:
        weights_vec.extend(w_mat.reshape(w_mat.size))

    weights_vec = np.array(weights_vec)
    return weights_vec


# Set weights of the neural network model
def set_weights(model, weights_vec):
    prev_weights = model.get_weights()
    # logging.info(prev_weights)
    new_weights = []
    start = 0

    for prev_w_mat in prev_weights:
        end = start + prev_w_mat.size
        new_w_mat = np.array(weights_vec[start:end]).reshape(prev_w_mat.shape)
        new_weights.append(new_w_mat)
        start = end

    model.set_weights(new_weights)


def test_weights_functions():
    ann = create_neural_net(10)
    prev_weights = ann.get_weights()
    vec = get_weights_vector(ann)
    # vec = [elem + 1 for elem in vec]

    ann2 = create_neural_net(10)
    set_weights(ann2, vec)
    post_weights = ann2.get_weights()

    for w_mat1, w_mat2 in zip(prev_weights, post_weights):
        assert np.array_equal(w_mat1, w_mat2)

    logging.info(prev_weights)
    logging.info(post_weights)


# %% [markdown]
# ## Main

# %%


def main():

    # -------------------------------------------
    # Setting parameters

    # Known paramaters are hx function (neural net), Q, R, w_init
    # No. of state variables = no. of weights in neural net
    # No. of measurement variables = D = 1 (y)

    dt = 0.01  # Setting learning rate
    n_samples = params.train_series_length  # Setting training series length

    # Create ANN, get its initial weights
    params.hxw_model = create_neural_net(X_train.shape[1])  # Create a neural net model
    w_init = get_weights_vector(params.hxw_model)  # Get weights from neural nets
    num_weights = w_init.shape[0]  # Number of weights inside the neural network

    # ---------------------------Filter Parameters-----------------------

    # -----------------UKF Parameter------------------
    P_init = 0.1 * np.eye(
        num_weights
    )  # Initial values of covariance matrix of state variables (MxM)
    Q = params.Q_var * np.eye(num_weights)  # Process noise covariance matrix (MxM)
    R = np.array([[params.R_var]])  # Measurement noise covariance matrix (DxD)

    sgd_ann = create_neural_net(X_train.shape[1])  # Create neural network model
    sgd_ann.set_weights(
        params.hxw_model.get_weights()
    )  # Set the weights for neural network (Same starting point as the UKF_ANN)

    ukf_ann = create_neural_net(X_train.shape[1])  # Create neural network for ukf
    testann = create_neural_net(X_train.shape[1])

    ukf_ann.set_weights(
        params.hxw_model.get_weights()
    )  # Set the ukf weighst same as sgd_nn (Same starting point as the UKF_ANN)

    z_true_series = y_train  # Set the test set as the training set
    num_iter = params.epochs * len(
        z_true_series
    )  # Initialize max_iteration: epochs * dataset_len

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)

    ukf_filter = create_ukf(Q, R, dt, w_init, P_init)  # Initialization of the UKF
    my_ukf = create_my_ukf(Q, R, dt, w_init, P_init)  # Initialiation of the SIF

    # Pre-allocate output variables
    ukf_w = np.zeros((num_weights, params.epochs))
    my_ukf_w = np.zeros((num_weights, params.epochs))
    ukf_train_accuracy = np.zeros(params.epochs)
    my_ukf_train_accuracy = np.zeros(params.epochs)
    sgd_train_mse = np.zeros(params.epochs)

    # -----------SIF Initalize Variables--------------
    x = w_init  # Weights of the neural network
    n = x.shape[0]  # Number of States
    m = z_true_series.shape[0]
    # delta = [[0.09], [9], [0.9]]
    delta = np.random.uniform(low=0.0009, high=0.9, size=(X_train.shape[0]))
    sat = np.zeros((m, m))
    C = np.ones((X_train.shape[0], n))
    P = P_init
    innovA = np.zeros((m, 1))

    N = len(x)
    w = np.zeros((x.shape))
    eta = 0.01
    x = get_weights_vector(ukf_ann)
    pdiff = np.zeros((num_iter, 1))
    # -------------------------------------------
    # Train SGD ANN (for comparison)
    logging.info("Training neural net with SGD")
    info_tracker = EpochInfoTracker()
    callbacks = [info_tracker]
    history = sgd_ann.fit(
        X_train, y_train, batch_size=1, epochs=1, verbose=3, callbacks=callbacks,
    )
    logging.info("Training SGD complete")
    # -------------------------------------------
    # Training loop with UKF
    out = StringIO()
    sifnn = []
    logging.info("Training neural net with UKF")
    t0 = time.time()
    epoch = 0
    # num_iter = 10 #hack
    minval = np.ones((num_iter, 1))
    aRate = 0.5

    # Epochs * len(y_train)
    for i in range(num_iter):
        # print("SHOUD", mean_squared_error(z_true_series, ukf_ann.predict(params.X_data)))
        idx = i % len(z_true_series)
        # logging.info(idx)
        if 0 == 0:
            if not params.train_ukf_ann:
                break
            # Checking the accuracy of the model
            preds_softmax = ukf_ann.predict(
                X_train
            )  # Model prediction (softmax format)
            z_true_series_accuracy = np.argmax(
                z_true_series, axis=1
            )  # Select the highest probability as the output
            preds_accuracy = np.argmax(
                preds_softmax, axis=1
            )  # Take the highest possibility as output among softmax output
            accuracy = accuracy_score(
                z_true_series_accuracy, preds_accuracy
            )  # Calculate the accuracy

            ukf_train_accuracy[epoch] = accuracy
            # my_ukf_train_accuracy[epoch] = accuracy
            sifnn.append(accuracy)
            print("The accuracy is: ", accuracy)
            if (accuracy >= 0.8) and (i > 1):
                thelast = i
                break
            # ukf_w[:, epoch] = x[:]
            # my_ukf_w[:, epoch] = x[:]

            epoch += 1

        # -----------------Genetic Algorithm------------
        accuracy_GA = []
        weights_GA = []
        for jj in range(100):
            weights_GA.append(get_weights_vector(ukf_ann))  # Store the weights vector
            accuracy_GA.append(accuracy)  # Store the accuracy values
            params.curr_idx = idx  # For use in hw() to fetch correct x_k sample
            z = z_true_series[idx]

            ##################################################
            # SIF Predicition Stage
            predict_genetic = ukf_ann.predict(
                X_train
            )  # Perform the prediction with given weights
            predict_genetic = np.max(
                predict_genetic, axis=1
            )  # Select the highest softmax output
            z_max = np.max(z_true_series, axis=1)

            innov = z_max - predict_genetic  # Set the innovation matrix
            x = get_weights_vector(ukf_ann) + aRate * np.sign(
                z_max[idx] - predict_genetic[idx]
            )
            # innov = z_true_series - np.dot(C,x)
            # print("Innov ", innov)

            ### Do something with delta

            # sat: saturation term
            ############## CHANGE #################
            for i in range(1, m):
                # innovA[i] = sum(innov[i])/len(innov[i])
                if (abs(innov[i]) / delta[i]) >= 1:
                    sat[i][i] = 1
                else:
                    sat[i][i] = abs(innov[i]) / delta[i]
            ######################

            pinvC = np.linalg.pinv(C)
            K = np.dot(pinvC, sat)
            x = np.asarray(
                [xx * random.uniform(0.001, 1) for xx in x]
            )  # Randomly initialize weights
            # print(x.shape, K.shape, innovA.shape, x.shape, np.dot(K, innovA).shape)
            was = np.reshape(np.dot(K, innov), x.shape)
            x = x + was  # NEED TO CHECK THIS LINE

            # print("Error " ,z_true_series[idx] - ukf_ann.predict(params.X_data)[idx])
            # print("ErrorA " ,z_true_series[idx] - sgd_ann.predict(params.X_data)[idx])

            set_weights(params.hxw_model, x)
            set_weights(ukf_ann, x)

            preds = ukf_ann.predict(X_train)
            preds = np.argmax(preds, axis=1)
            accuracy = accuracy_score(z_true_series_accuracy, preds)

            # print("Accuracy: ", accuracy)
            minval[idx] = accuracy
            pdiff[idx] = accuracy
            # if accuracy <= 0.03:
            #   break
            # ukf_filter.x = x
            # print(type(preds))
            accuracy_GA.append(accuracy)
            weights_GA.append(x)
            # print(accuracy_GA)

            # Genetic Algorithm
            if jj == 99:
                # print(accuracy_GA[accuracy_GA.index(min(accuracy_GA))])

                set_weights(ukf_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])
                preds = ukf_ann.predict(X_train)
                #     print("Last Set: ", mean_squared_error(z_true_series, preds))
                set_weights(ukf_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])
        #     print("jjj",mean_squared_error(z_true_series, ukf_ann.predict(params.X_data) ))

    time_to_train = time.time() - t0
    logging.info(
        "Training complete. time_to_train = {:.2f} sec, {:.2f} min".format(
            time_to_train, time_to_train / 60
        )
    )


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This line disables GPU
    main()

# %% [markdown]
# ## Result Analysis

# %%
# -------------------------------------------
# Results analysis

# Visualize evolution of ANN weights

# Visualize error curve (SGD vs UKF)
x_var = range(thelast + 1)
hist = history.history["loss"]
ukf_train_mse = np.array(sifnn)
# utility.plot(x_var, hist, xlabel='Epoch',
#            label='SGD ANN training history (MSE)')
utility.plot(
    x_var, ukf_train_mse, new_figure=False, label="SIF ANN training history (MSE)"
)

# True test series vs. ANN pred vs, UKF pred
logging.info("Evaluating and visualizing neural net predictions")
evaluate_neural_nets(
    sgd_ann, ukf_ann, window, use_train_series=True, train_series=X_series
)
evaluate_neural_nets(sgd_ann, ukf_ann, window)

utility.save_all_figures("output")
plt.show()

print("The Min MSE is ", min(minval), " vs ", hist[-1])
print("Total amount of epochs for SIF: ", epoch)

# %% [markdown]
# ## SIF (Step by step)

# %%
dt = 0.01  # Setting learning rate
n_samples = params.train_series_length  # Setting batch size

# Create ANN, get its initial weights
params.hxw_model = create_neural_net(X_train.shape[1])  # Create a neural net model
w_init = get_weights_vector(params.hxw_model)  # Get weights from neural nets
num_weights = w_init.shape[0]  # Number of weights inside the neural network

ukf_ann = create_neural_net(X_train.shape[1])

z_true_series = y_train

num_iter = params.epochs * len(z_true_series)


# %%
# -----------SIF Initalize Variables--------------
x = w_init
n = x.shape[0]  # Number of States
m = z_true_series.shape[0]
# delta = [[0.09], [9], [0.9]]
delta = np.random.uniform(low=0.0009, high=0.9, size=(X_train.shape[0]))
sat = np.zeros((m, m))
C = np.ones((X_train.shape[0], n))
# P = P_init
innovA = np.zeros((m, 1))

N = len(x)
w = np.zeros((x.shape))
eta = 0.01
x = get_weights_vector(ukf_ann)
pdiff = np.zeros((num_iter, 1))


# %%
logging.info("Training neural net with SGD")
info_tracker = EpochInfoTracker()
callbacks = [info_tracker]
history = ukf_ann.fit(
    X_train, y_train, batch_size=1, epochs=1, verbose=3, callbacks=callbacks,
)


# %%
preds_softmax = ukf_ann.predict(X_train)


z_true_series_accuracy = np.argmax(z_true_series, axis=1)
preds_accuracy = np.argmax(
    preds_softmax, axis=1
)  # Take the highest possibility as output among softmax output
accuracy = accuracy_score(z_true_series_accuracy, preds_accuracy)
print(accuracy)

# %% [markdown]
# ## Prediction Stage
#
# predicted innovation $ \hat{z}_{k+1|k} $: pred_innov
#
#

# %%
preds_softmax = ukf_ann.predict(X_train)
predict_genetic = np.max(predict_genetic, axis=1)  # Select the highest softmax output
z_max = np.max(z_true_series, axis=1)

pred_innov = z_max - predict_genetic

