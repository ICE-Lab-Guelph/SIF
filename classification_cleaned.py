# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Tuning A Neural Network Using SIF vs KF (Classification Task)

# %%
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
# params.epochs = 1200
# params.train_series_length = X_train.shape[0]
# params.test_series_length = X_test.shape[0]

# params.window_size = 12    #
# params.ukf_dt = 0.1
# params.alpha, params.beta, params.kappa = 1, 2, 1  # Worked well
# params.Q_var = 0.001
# params.R_var = 0.001


# To make training data and related variables accessible across functions
# params.train_ukf_ann = True
# params.X_data = None
# params.y_data = None
# params.hxw_model = None
# params.curr_idx = 0

"""
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

"""
# Create a simple feedforward neural network
def create_neural_net(M):
    """
    M: input dimension of the neural network
    """

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


"""
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
"""

# %% [markdown]
# ## SIF (Step by step)
# %% [markdown]
# Initialization of neural networks

# %%
n_samples = X_train.shape[1]  # Setting training set length

# Create ANN, get its initial weights
sif_ann = create_neural_net(X_train.shape[1])  # Create a neural net model
w_init = get_weights_vector(sif_ann)  # Get weights from neural nets
num_weights = w_init.shape[0]  # Number of weights inside the neural network

z_true_series = y_train  # Set the labels as your true series

# %% [markdown]
# Getting and setting initial SIF parameters

# %%
# -----------Initalize & Pre-allocate SIF Variables--------------
x = w_init  # Set the initial NN weights
n = x.shape[0]  # Number of States (11)
m = z_true_series.shape[0]  # Number of measurements (105 = )

delta = np.random.uniform(
    low=0.0009, high=0.9, size=(X_train.shape[0])
)  # Assign delta values for each data point
sat = np.zeros((m, m))  # Pre-allocation of saturation matrix
C = np.ones((X_train.shape[0], n))  # Pre-allocation of measurement matrix

N = len(x)  #
w = np.zeros((x.shape))
eta = 0.01
x = get_weights_vector(sif_ann)

# %% [markdown]
# ## Prediction Stage
#
# Eq.(3.3):  $$ \hat{z}_{k+1|k}  =  z_{k+1|k}  - h  \hat{x}_{k+1|k} $$
#
# <br></br>
#
# * Predicted innovation $ \hat{z}_{k+1|k} $: pred_innov
#
# * True values $ z_{k+1|k} $:  z_true_series
#
# * Measurement matrix $ \hat{x}_{k+1|k} $: prediction
#
# * Nonlinear measuement function $h()$: sif_ann.predict()
#

# %%
preds_softmax = sif_ann.predict(X_train)  # NN_model (weights)

masked_output = (
    z_true_series.max(axis=1, keepdims=1) == z_true_series
)  # Initialize the masking array which returns the true class

predict_output = preds_softmax[
    masked_output
]  # Slice the softmax value of the true class
z_output = z_true_series[masked_output]  # Slice the true class vlaues

pred_innov = z_output - predict_output  # Compute the predicted innovation matrix
print(
    "Predicted innovation's shape: ", pred_innov.shape
)  # Print the predicted innovation matrix's shape


accuracy = accuracy_score(
    z_true_series.argmax(axis=1), preds_softmax.argmax(axis=1)
)  # Compute the accuracy
print("Accuracy: ", accuracy)


# %%
aRate = 0.05

x = get_weights_vector(
    sif_ann
) + aRate * np.sign(  # Slighly change the weights, so the SIF won't get stuck on the same value.
    z_output[0] - predict_output[0]
)
print("Calculated weights: ", x)

# %% [markdown]
# ### Computation of the SIF Gain ($K_{k+1}$):
# <br></br>
# Eq.(3.4) : $$K_{k+1} = C^{+} \overline{sat} (|\hat{z}_{k+1|k}| / \delta)$$
#
# * $\overline{sat}$ refers to the diagonal of the saturation term, sat refers to the saturation of a value (yields a result between 1 and -1)
#
# * Note that $C^{+}$ refers to the pseudoinverse of the measurement matrix

# %%
delta = np.random.uniform(low=0.0009, high=0.9, size=(X_train.shape[0]))

m = 105  # Length of the training samples
# -------------saturation computation----------------
for i in range(1, m):
    if (
        abs(pred_innov[i]) / delta[i]
    ) >= 1:  # If the value is greater than 1 keep it as 1
        sat[i][i] = 1

    elif (
        abs(pred_innov[i]) / delta[i]
    ) <= -1:  # If the value is lower than -1 keep it as -1
        sat[i][i] = -1

    else:
        sat[i][i] = (
            abs(pred_innov[i]) / delta[i]
        )  # If the value is between -1 and 1 keep it as it is
# -----------saturation computation-----------------

pinvC = np.linalg.pinv(C)  # Pseudo inverse of C (105,11)
K = np.dot(pinvC, sat)  # Calculation of SIF gain

print("SIF gain shape: ", K.shape)  # SIF_gain (4,11)

# %% [markdown]
# Computation of State Estimate ($\hat{x}_{k+1|k+1}$):
#
# Eq. (3.5): $$\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} +  (K_{k+1})(\hat{z}_{k+1|k})$$
#
#

# %%
w_init = get_weights_vector(sif_ann)  # Get the initial neuron weights
x = w_init  # Set them as our measurements

x = x + np.dot(K, pred_innov)  # Compute the state estimate

print("Previous weights: \n", w_init)
print("Updated weights: \n", x)

# %% [markdown]
# ## Genetic Algorithm (without crossover & mutation)

# %%
# -----------------Genetic Algorithm------------
accuracy_GA = []  # Accuracy values of each individual
weights_GA = []  # Weights of each individual

# Genetic Algorithm loop
# In each iteration a random weights are assigned and evaluated based on their accuracy.
epoch = 0
for jj in range(100):

    weights_GA.append(get_weights_vector(sif_ann))  # Store the weights vector
    accuracy_GA.append(accuracy)  # Store the accuracy values
    ##################################################

    # SIF Predicition Stage
    preds_softmax = sif_ann.predict(X_train)  # NN_model (weights)

    masked_output = (
        z_true_series.max(axis=1, keepdims=1) == z_true_series
    )  # Initialize the masking array which returns the true class

    predict_output = preds_softmax[
        masked_output
    ]  # Slice the softmax value of the true class
    z_output = z_true_series[masked_output]  # Slice the true class vlaues

    pred_innov = z_output - predict_output  # Compute the predicted innovation matrix

    # -------------saturation computation----------------
    for i in range(1, m):
        if (abs(pred_innov[i]) / delta[i]) >= 1:  # If the value
            sat[i][i] = 1

        elif (abs(pred_innov[i]) / delta[i]) <= -1:
            sat[i][i] = -1

        else:
            sat[i][i] = abs(pred_innov[i]) / delta[i]
    # -----------saturation computation-----------------

    pinvC = np.linalg.pinv(C)
    K = np.dot(pinvC, sat)
    x = np.asarray(
        [xx * random.uniform(0.001, 1) for xx in x]
    )  # Randomly initialize weights

    was = np.reshape(np.dot(K, pred_innov), x.shape)
    x = x + was  # NEED TO CHECK THIS LINE

    set_weights(sif_ann, x)

    preds = sif_ann.predict(X_train)
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(z_true_series.argmax(axis=1), preds)
    print(f"NN weights no:{jj}, accuracy: {accuracy}")

    accuracy_GA.append(accuracy)
    weights_GA.append(x)
    # print(accuracy_GA)

    # Genetic Algorithm
    if jj == 99:
        # print(accuracy_GA[accuracy_GA.index(min(accuracy_GA))])

        set_weights(sif_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])
        preds = sif_ann.predict(X_train)
        set_weights(sif_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])


# %%
C = np.ones((X_train.shape[0], n))
C.shape


# %%
print("First Accuracy: ", accuracy_GA[0])
print("After 100 random weights and SIF approximation...")
print("Best Accuracy: ", np.max(accuracy_GA))

# %% [markdown]
# ## Main

# %%
def SIF_ANN(w_init, epoch, max_epoch=1000):
    # -------------------------------------------
    # Setting parameters
    n_samples = X_train.shape[1]  # Setting training series length

    # Create ANN, get its initial weights
    sif_ann = create_neural_net(X_train.shape[1])  # Create a neural net model
    num_weights = w_init.shape[0]  # Number of weights inside the neural network

    # ---------------------------Filter Parameters-----------------------
    z_true_series = y_train  # Set the test set as the training set
    num_iter = max_epoch * len(
        z_true_series
    )  # Initialize max_iteration: epochs * dataset_len

    # -----------SIF Initalize Variables--------------
    x = w_init  # Weights of the neural network
    n = x.shape[0]  # Number of States: 11
    m = z_true_series.shape[0]  # Measurement matrix: 105

    delta = np.random.uniform(low=0.0009, high=0.9, size=(X_train.shape[0]))
    sat = np.zeros((m, m))
    C = np.ones((X_train.shape[0], n))
    P = 0.1 * np.eye(num_weights)
    innovA = np.zeros((m, 1))

    N = len(x)
    w = np.zeros((x.shape))
    x = get_weights_vector(sif_ann)
    # Training loop with UKF
    aRate = 0.05

    # Epochs * len(y_train)
    for i in range(num_iter):
        idx = i % len(z_true_series)

        # Checking the accuracy of the model
        preds_softmax = sif_ann.predict(X_train)  # Model prediction (softmax format)
        z_true_series_accuracy = np.argmax(
            z_true_series, axis=1
        )  # Select the highest probability as the output
        preds_accuracy = np.argmax(
            preds_softmax, axis=1
        )  # Take the highest possibility as output among softmax output
        accuracy = accuracy_score(
            z_true_series_accuracy, preds_accuracy
        )  # Calculate the accuracy

        sif_ann_accuracy.append(accuracy)
        print("The accuracy is: ", accuracy)
        if (accuracy >= 0.85) and (i > 1):
            thelast = i
            break

        epoch += 1

        # -----------------Genetic Algorithm------------
        accuracy_GA = []  # Accuracy values of each individual
        weights_GA = []  # Weights of each individual

        # Genetic Algorithm loop
        # In each iteration a random weights are assigned and evaluated based on their accuracy.
        for jj in range(100):

            weights_GA.append(get_weights_vector(sif_ann))  # Store the weights vector
            accuracy_GA.append(accuracy)  # Store the accuracy values
            ##################################################

            # SIF Predicition Stage
            preds_softmax = sif_ann.predict(X_train)  # NN_model (weights)

            masked_output = (
                z_true_series.max(axis=1, keepdims=1) == z_true_series
            )  # Initialize the masking array which returns the true class

            predict_output = preds_softmax[
                masked_output
            ]  # Slice the softmax value of the true class
            z_output = z_true_series[masked_output]  # Slice the true class vlaues

            pred_innov = (
                z_output - predict_output
            )  # Compute the predicted innovation matrix

            # -------------saturation computation----------------
            for i in range(1, m):
                if (abs(pred_innov[i]) / delta[i]) >= 1:  # If the value
                    sat[i][i] = 1

                elif (abs(pred_innov[i]) / delta[i]) <= -1:
                    sat[i][i] = -1

                else:
                    sat[i][i] = abs(pred_innov[i]) / delta[i]
            # -----------saturation computation-----------------

            pinvC = np.linalg.pinv(C)
            K = np.dot(pinvC, sat)
            x = np.asarray(
                [xx * random.uniform(0.001, 1) for xx in x]
            )  # Randomly initialize weights

            was = np.reshape(np.dot(K, pred_innov), x.shape)
            x = x + was  # NEED TO CHECK THIS LINE

            set_weights(sif_ann, x)

            preds = sif_ann.predict(X_train)
            accuracy = accuracy_score(
                z_true_series.argmax(axis=1), preds.argmax(axis=1)
            )

            accuracy_GA.append(accuracy)
            weights_GA.append(x)

            # Pick the best individual
            if jj == 99:
                set_weights(sif_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])
                preds = sif_ann.predict(X_train)
                set_weights(sif_ann, weights_GA[accuracy_GA.index(max(accuracy_GA))])
        """
        time_to_train = time.time() - t0
        logging.info(
            "Training complete. time_to_train = {:.2f} sec, {:.2f} min".format(
                time_to_train, time_to_train / 60
            )
        )
        """


nn = create_neural_net(X_train.shape[1])
w_init = get_weights_vector(nn)
sif_ann_accuracy = []
epoch = 0

SIF_ANN(w_init, epoch=epoch)
# %% [markdown]
# ## Result Analysis

# %%
# ---------------Results analysis--------------

# Visualize evolution of ANN weights

# Visualize error curve (SGD vs UKF)
x_var = range(thelast + 1)  # Get the last epoch's number
hist = history.history["loss"]  # Get the history of the NN with SGD
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

