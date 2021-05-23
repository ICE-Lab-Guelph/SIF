# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Tuning A Neural Network Using SIF vs KF (Regression Task)

# %%
# Installing necessary libraries
# get_ipython().system("pip install filterpy")

# Importing global modules
from pprint import pformat
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
from sklearn.metrics import mean_squared_error
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

# %% [markdown]
#

# %%
# Tracking of weights by epoch
class EpochInfoTracker(Callback):
    def __init__(self):
        self.weights_history = []  # Tracking the weights in each epochs

    def on_epoch_end(self, epoch, logs=None):
        weights_vec = get_weights_vector(self.model)
        self.weights_history.append(weights_vec)


# Class for keeping the necessary parameters
class Params:
    pass


# %% [markdown]
#

# %%
# --------------Initialization of the parameters------------
params = Params()
params.epochs = 1200
params.train_series_length = 500
params.test_series_length = 1000
params.mg_tau = 30
params.window_size = 12  # M
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


def measurement_func(w, x):
    hxw_model = params.hxw_model
    qq = np.asarray(w)
    ww = np.reshape(qq, (13))
    set_weights(hxw_model, ww)
    # Reshape needed to feed x as 1 sample to ANN model
    hxw = hxw_model.predict(x.reshape(1, len(x)))
    hxw = hxw.flatten()  # Flatten to make shape = (1,)
    return hxw


def fw(w, dt=None):
    return w  # Identity


def hw(w):
    x = params.X_data[params.curr_idx]
    hxw = measurement_func(w, x)
    return hxw


# Create ukf using pykalman
def create_ukf(Q, R, dt, w_init, P_init):
    M = w_init.shape[0]

    points = MerweScaledSigmaPoints(M, params.alpha, params.beta, params.kappa)

    ukf = UnscentedKalmanFilter(dim_x=M, dim_z=1, dt=dt, fx=fw, hx=hw, points=points)
    ukf.x = w_init
    ukf.P = P_init
    ukf.R = R
    ukf.Q = Q

    return ukf


# Create ukf using ukf.py
def create_my_ukf(Q, R, dt, w_init, P_init):
    my_ukf = ukf.UnscentedKalmanFilter(
        fw, hw, R, Q, w_init, P_init, params.alpha, params.beta, params.kappa
    )
    return my_ukf


# Reshape the dataset as: (batchsize, window_size)
def prepare_dataset(series, M, stride):
    X, y = [], []
    for i in range(0, len(series) - M - 1, stride):
        window = series[i : (i + M)]  #
        X.append(window)
        y.append(series[i + M])
    return np.array(X), np.array(y)


def evaluate_neural_nets(
    sgd_ann, ukf_ann, window, use_train_series=False, train_series=None
):
    if use_train_series:
        X_data, y_data = params.X_data, params.y_data
        series = train_series
        sample_len = params.train_series_length
        title = "Train series (true vs. predicted)"
    else:
        sample_len = params.test_series_length
        series = utility.mackey_glass(sample_len=sample_len, tau=params.mg_tau)
        series = np.array(series[0]).reshape((sample_len))
        X_data, y_data = prepare_dataset(series, window, stride=1)
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
        mse = mean_squared_error(y_data, preds)
        print("The Test MSE is: ", mse)

    # utility.plot(range(sample_len), y_self_pred_series, new_figure=False,
    #              label='Predicted test series (rolling prediction: no true vals used)')


# Create a simple feedforward neural network
def create_neural_net(M):

    # Build a simple neural network
    ann = Sequential()
    ann.add(Dense(1, input_dim=M, activation="tanh"))
    ann.compile(optimizer="sgd", loss="mse")

    # Print out the summary of the model
    ann.summary()

    return ann


def get_weights_vector(model):
    weights = model.get_weights()
    # logging.info(weights)
    weights_vec = []
    for w_mat in weights:
        weights_vec.extend(w_mat.reshape(w_mat.size))

    weights_vec = np.array(weights_vec)
    return weights_vec


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
# Initialization of the functions and essential parameters before exxecution of the algorithm.

# %%
def main():
    # utility.setup_logging('output')
    # logging.info('Experiment parameters below')
    # logging.info('\n{}'.format(pformat(params.__dict__)))

    # test_weights_functions()
    # assert False

    # -------------------------------------------
    # Setting parameters

    # Known paramaters are hx function (neural net), Q, R, w_init
    # No. of state variables = no. of weights in neural net
    # No. of measurement variables = D = 1 (y)

    window = params.window_size  # Setting window size
    dt = 0.01  # Setting learning rate
    n_samples = params.train_series_length  # Setting batch size

    # -------------------------------------------
    # Generating data
    X_series = utility.mackey_glass(  # Generating the dataset (Mackey Glass)
        sample_len=n_samples, tau=params.mg_tau, n_samples=window
    )
    X_series = np.array(X_series[0]).reshape((n_samples))

    params.X_data, params.y_data = prepare_dataset(
        X_series, window, stride=1
    )  # Reshaping dataset for a regression task
    # params.X_data, params.y_data = xx , yy
    # Create ANN, get its initial weights
    params.hxw_model = create_neural_net(window)  # Create neural net model
    w_init = get_weights_vector(params.hxw_model)  # Get weights from neural nets
    num_weights = w_init.shape[0]

    # ---------------------------Filter Parameters-----------------------

    # -----------------UKF Parameter------------------
    # Initial values of covariance matrix of state variables (MxM)
    P_init = 0.1 * np.eye(num_weights)
    # Process noise covariance matrix (MxM)
    Q = params.Q_var * np.eye(num_weights)
    R = np.array([[params.R_var]])  # Measurement noise covariance matrix (DxD)

    sgd_ann = create_neural_net(window)
    # Same starting point as the UKF_ANN
    sgd_ann.set_weights(params.hxw_model.get_weights())

    ukf_ann = create_neural_net(window)
    testann = create_neural_net(window)

    # Same starting point as the UKF_ANN
    ukf_ann.set_weights(params.hxw_model.get_weights())

    z_true_series = params.y_data
    num_iter = params.epochs * len(z_true_series)

    # 2 Kalman filter implementations to compare (from filterpy and my custom impl)
    ukf_filter = create_ukf(Q, R, dt, w_init, P_init)  # Initialization of the UKF
    my_ukf = create_my_ukf(Q, R, dt, w_init, P_init)  # Initialiation of the SIF

    # Pre-allocate output variables
    ukf_w = np.zeros((num_weights, params.epochs))
    my_ukf_w = np.zeros((num_weights, params.epochs))
    ukf_train_mse = np.zeros(params.epochs)
    my_ukf_train_mse = np.zeros(params.epochs)
    sgd_train_mse = np.zeros(params.epochs)

    # SIF Initalize Variables
    x = w_init
    n = x.shape[0]  # Number of States
    m = z_true_series.shape[0]
    # delta = [[0.09], [9], [0.9]]
    delta = np.random.uniform(low=0.0009, high=0.9, size=(487,))
    sat = np.zeros((m, m))
    C = np.ones((487, 13))
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
        params.X_data,
        params.y_data,
        batch_size=1,
        epochs=1,
        verbose=3,
        callbacks=callbacks,
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
    for i in range(num_iter):
        # print("SHOUD", mean_squared_error(z_true_series, ukf_ann.predict(params.X_data)))
        idx = i % len(z_true_series)
        # logging.info(idx)
        if 0 == 0:
            if not params.train_ukf_ann:
                break

            preds = ukf_ann.predict(params.X_data)
            mse = mean_squared_error(z_true_series, preds)
            ukf_train_mse[epoch] = mse
            my_ukf_train_mse[epoch] = mse
            sifnn.append(mse)
            print("The MSE is: ", mse)
            if mse <= 0.01 and i > 10:
                thelast = i
                break
            # ukf_w[:, epoch] = x[:]
            # my_ukf_w[:, epoch] = x[:]

            epoch += 1
            # logging.info('Epoch: {} / {}'.format(epoch, params.epochs))

        # -----------------Genetic Algorithm------------
        geneticMSE = []
        geneticWeights = []
        for jj in range(100):
            geneticWeights.append(
                get_weights_vector(ukf_ann)
            )  # Store the weights vector
            geneticMSE.append(mse)  # Store the MSE values
            params.curr_idx = (
                idx  # For use in hw() to fetch correct x_k sample            #
            )
            z = z_true_series[idx]

            ##################################################
            # SIF Predicition Stage
            # Learning Rate at 0.01

            # error = mean_squared_error(z_true_series, preds)
            # gradient = x.T * error / preds.shape[0]
            # w = w - eta * gradient

            # ukf_filter.predict()

            # 1*get_weights_vector(ukf_ann) + 0.01#ukf_filter.x # + 0.01 * w

            # print("WW  ", z_true_series.shape, x.shape)
            # pdb.set_trace()
            innov = z_true_series - np.reshape(
                ukf_ann.predict(params.X_data), z_true_series.shape
            )
            x = get_weights_vector(ukf_ann) + aRate * np.sign(
                z_true_series[idx] - ukf_ann.predict(params.X_data)[idx]
            )
            # innov = z_true_series - np.dot(C,x)
            # print("Innov ", innov)

            ### Do something with delta

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
            x = np.asarray([xx * random.uniform(0.001, 1) for xx in x])
            # print(x.shape, K.shape, innovA.shape, x.shape, np.dot(K, innovA).shape)
            was = np.reshape(np.dot(K, innov), x.shape)
            x = x + was  # NEED TO CHECK THIS LINE

            # print("Error " ,z_true_series[idx] - ukf_ann.predict(params.X_data)[idx])
            # print("ErrorA " ,z_true_series[idx] - sgd_ann.predict(params.X_data)[idx])

            set_weights(params.hxw_model, x)
            set_weights(ukf_ann, x)
            preds = ukf_ann.predict(params.X_data)
            mse = mean_squared_error(z_true_series, preds)
            print("MSEN: ", mse)
            minval[idx] = mse
            pdiff[idx] = mse
            # if mse <= 0.03:
            #   break
            # ukf_filter.x = x
            print(type(preds))
            geneticMSE.append(mse)
            geneticWeights.append(x)
            print(geneticMSE)

            # Genetic Algorithm
            if jj == 99:
                print(geneticMSE[geneticMSE.index(min(geneticMSE))])

                set_weights(ukf_ann, geneticWeights[geneticMSE.index(min(geneticMSE))])
                preds = ukf_ann.predict(params.X_data)
                #     print("Last Set: ", mean_squared_error(z_true_series, preds))
                set_weights(ukf_ann, geneticWeights[geneticMSE.index(min(geneticMSE))])
        #     print("jjj",mean_squared_error(z_true_series, ukf_ann.predict(params.X_data) ))

    time_to_train = time.time() - t0
    logging.info(
        "Training complete. time_to_train = {:.2f} sec, {:.2f} min".format(
            time_to_train, time_to_train / 60
        )
    )

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


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This line disables GPU
    main()


# %%
xx, yy = params.X_data, params.y_data


# %%
get_ipython().system("zip -r output.zip output/")

