#%%
"""
Contains a class for EKF-training a feedforward neural-network.
This is primarily to demonstrate the advantages of EKF-training.
See the class docstrings for more details.
This module also includes a function for loading stored KNN objects.
"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.linalg import block_diag
from time import time
import pickle

##########

def load_knn(filename):
    """
    Loads a stored KNN object saved with the string filename.
    Returns the loaded object.
    """
    if not isinstance(filename, str):
        raise ValueError("The filename must be a string.")
    if filename[-4:] != '.knn':
        filename = filename + '.knn'
    with open(filename, 'rb') as input:
        W, neuron, P = pickle.load(input)
    obj = KNN(W[0].shape[1]-1, W[1].shape[0], W[0].shape[0], neuron)
    obj.W, obj.P = W, P
    return obj

##########

class KNN:
    """
    Class for a feedforward neural network (NN). Currently only handles 1 hidden-layer,
    is always fully-connected, and uses the same activation function type for every neuron.
    The NN can be trained by extended kalman filter (EKF) or stochastic gradient descent (SGD).
    Use the train function to train the NN, the feedforward function to compute the NN output,
    and the classify function to round a feedforward to the nearest class values. A save function
    is also provided to store a KNN object in the working directory.
    """
    def __init__(self, nu, ny, nl, neuron, sprW=0.2):
        """
            nu: dimensionality of input; positive integer
            ny: dimensionality of output; positive integer
            nl: number of hidden-layer neurons; positive integer
        neuron: activation function type; 'logistic', 'tanh', or 'relu'
          sprW: spread of initial randomly sampled synapse weights; float scalar
        """
        # Function dimensionalities
        self.nu = int(nu)
        self.ny = int(ny)
        self.nl = int(nl)

        # Neuron type
        if neuron == 'logistic':
            self.sig = lambda V: (1 + np.exp(-V))**-1
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = lambda V: np.tanh(V)
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.float64(sigV > 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        self.neuron = neuron

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))     # Number of weights
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

####

    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.
        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)

####

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l
        """
        U = np.float64(U)
        if U.ndim == 1 and len(U) > self.nu: U = U[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], U))
        h = self._affine_dot(self.W[1], l)
        if get_l: return h, l
        return h

####

    def classify(self, U, high, low=0):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        For each associated output, the closest integer between high
        and low is returned as a (m by ny) classification matrix.
        Basically, your training data should be (u, int_between_high_low).
        """
        return np.int64(np.clip(np.round(self.feedforward(U), 0), low, high))

####

    def train(self, nepochs, U, Y, method, P=None, Q=None, R=None, step=1, dtol=-1, dslew=1, pulse_T=-1, gradient_testing=False):
        """
        nepochs: number of epochs (presentations of the training data); integer
              U: input training data; float array m samples by nu inputs
              Y: output training data; float array m samples by ny outputs
         method: extended kalman filter ('ekf') or stochastic gradient descent ('sgd')
              P: initial weight covariance for ekf; float scalar or (nW by nW) posdef array
              Q: process covariance for ekf; float scalar or (nW by nW) semiposdef array
              R: data covariance for ekf; float scalar or (ny by ny) posdef array
           step: step-size scaling; float scalar
           dtol: finish when RMS error avg change is <dtol (or nepochs exceeded); float scalar
          dslew: how many deltas over which to examine average RMS change; integer
        pulse_T: number of seconds between displaying current training status; float
        If method is 'sgd' then P, Q, and R are unused, so carefully choose step.
        If method is 'ekf' then step=1 is "optimal", R must be specified, and:
            P is None: P = self.P if self.P has been created by previous training
            Q is None: Q = 0
        If P, Q, or R are given as scalars, they will scale an identity matrix.
        Set pulse_T to -1 (default) to suppress training status display.
        Returns a list of the RMS errors at every epoch and a list of the covariance traces
        at every iteration. The covariance trace list will be empty if using sgd.
        """
        # Verify data
        self.gradient_testing = gradient_testing
        U = np.float64(U)   # Input data
        Y = np.float64(Y)   # Output data
        if len(U) != len(Y):
            raise ValueError("Number of input data points must match number of output data points.")
        if (U.ndim == 1 and self.nu != 1) or (U.ndim != 1 and U.shape[-1] != self.nu):
            raise ValueError("Shape of U must be (m by nu).")
        if (Y.ndim == 1 and self.ny != 1) or (Y.ndim != 1 and Y.shape[-1] != self.ny):
            raise ValueError("Shape of Y must be (m by ny).")
        if Y.ndim == 1 and len(Y) > self.ny: Y = Y[:, np.newaxis]

        # Set-up
        # if method == 'ekf':
        if method == 'sif' or method == "ekf":
            # self.update = self._ekf
            if method == 'sif':
                self.update = self._sif
            elif method == 'ekf':
                self.update = self._ekf

            if P is None:
                if self.P is None:
                    raise ValueError("Initial P not specified.")
            elif np.isscalar(P):
                self.P = P*np.eye(self.nW)  # Initial covariance
            else:
                if np.shape(P) != (self.nW, self.nW):
                    raise ValueError("P must be a float scalar or (nW by nW) array.")
                self.P = np.float64(P)

            if Q is None:
                self.Q = np.zeros((self.nW, self.nW))
            elif np.isscalar(Q):
                self.Q = Q*np.eye(self.nW)
            else:
                if np.shape(Q) != (self.nW, self.nW):
                    raise ValueError("Q must be a float scalar or (nW by nW) array.")
                self.Q = np.float64(Q)
            if np.any(self.Q): self.Q_nonzero = True
            else: self.Q_nonzero = False

            if R is None:
                raise ValueError("R must be specified for EKF training.")
            elif np.isscalar(R):
                self.R = R*np.eye(self.ny)
            else:
                if np.shape(R) != (self.ny, self.ny):
                    raise ValueError("R must be a float scalar or (ny by ny) array.")
                self.R = np.float64(R)
            if npl.matrix_rank(self.R) != len(self.R):
                raise ValueError("R must be positive definite.")

        elif method == 'sgd':
            self.update = self._sgd
        else:
            raise ValueError("The method argument must be either 'ekf' or 'sgd'.")
        last_pulse = 0
        self.RMS = []
        self.RMS_batch = []
        self.W_hist = np.hstack((np.asarray(self.W[0]).ravel(), np.asarray(self.W[1]).ravel()))
        self.delta_vals = []
        self.nn_outputs = []
        trcov = []


        # Shuffle data between epochs
        # print(f"Training {method}...")
        for epoch in range(nepochs):

            rand_idx = np.random.permutation(len(U))
            U_shuffled = U[rand_idx]
            Y_shuffled = Y[rand_idx]
            self.RMS.append(self.compute_rms(U, Y))

            # Check for convergence
            if len(self.RMS) > dslew and abs(self.RMS[-1] - self.RMS[-1-dslew])/dslew < dtol:
                # print("\nConverged after {} epochs!\n\n".format(epoch+1))
                return self.RMS, trcov

            # Train
            for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

                # Forward propagation
                h, l = self.feedforward(u, get_l=True)
                self.nn_outputs.append(h)
                # Do the learning
                self.update(u, y, h, l, step)
                if method == 'sif' or method == 'ekf': trcov.append(np.trace(self.P))
                

                # Heartbeat
                if (pulse_T >= 0 and time()-last_pulse > pulse_T) or (epoch == nepochs-1 and i == len(U)-1):
                    # print("------------------")
                    # print("  Epoch: {}%".format(int(100*(epoch+1)/nepochs)))
                    # print("   Iter: {}%".format(int(100*(i+1)/len(U))))
                    # print("   RMSE: {}".format(np.round(self.RMS[-1], 6)))
                    # if method == 'ekf': print("tr(Cov): {}".format(np.round(trcov[-1], 6)))
                    # print("------------------")
                    last_pulse = time()
                
                # Compute RMS and save delta
                self.RMS_batch.append(self.compute_rms(U, Y))
                if method == 'sif': self.delta_vals.append(self.delta[0])

                    

            flat_weights = np.hstack((np.asarray(self.W[0]).ravel(), np.asarray(self.W[1]).ravel()))
            self.W_hist = np.vstack((self.W_hist, flat_weights))

        # print(f"\n{method}Training complete!\n\n")
        self.RMS.append(self.compute_rms(U, Y))
        return self.RMS, trcov

####

    def _ekf(self, u, y, h, l, step):

        # Compute NN jacobian
        # H: Jacobian matrix with partial derivatives
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                       block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = step*K.dot(y-h)

        if self.gradient_testing:
            W_next = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
            self.W[0][0,0], self.W[0][1,0] = W_next[0,0], W_next[1,0] 
        else:
            self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
            self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        
        self.P = self.P - np.dot(K, H.dot(self.P))
        if self.Q_nonzero: self.P = self.P + self.Q


    def _sif(self, u, y, h, l, step):
        
        
        #======Prediction Stage======#
        # 'priori' state estimates (before the fact)
        flat_weights = np.hstack((np.asarray(self.W[0]).ravel(), np.asarray(self.W[1]).ravel()))
        m = len(flat_weights)
        #Predicted innovation (3.13)
        z = y - h
        #======Prediction Stage======#

        # e_k = h - np.asarray(self.nn_outputs).mean()
        # self.R = (e_k @ e_k.T).reshape(1,1)

        #======Update Stage===========#
        # Compute NN jacobian
        # H: Jacobian matrix with partial derivatives 
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                        block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))
        
        self.P = H @ self.P @ H.T + self.Q

        #------------- saturation computation ----------------
        # sat = np.zeros((m,1)) 
        # delta = np.random.uniform(low=0.0009, high=0.009, size=(m,1))
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        self.delta = S @ np.linalg.pinv(S - self.R) @ np.abs(z)

        # for i in range(m):
        #     if (abs(flat_weights[i]) / delta) >= 1:        # If the value is greater than 1 keep it as 1
        #         sat[i,:] = 1

        #     elif (abs(flat_weights[i]) / delta) <= -1:     # If the value is lower than -1 keep it as -1
        #         sat[i,:] = -1 

        #     else:
        #         sat[i,:] = abs(flat_weights[i]) / delta   # If the value is between -1 and 1 keep it as it is
        #------------- saturation computation-----------------
        
        # ESIF Gain (3.14)
        K = np.linalg.pinv(H) * np.abs(z) * (1/self.delta)
        # 'posteriori' state estimates (after the fact)
        dW = step * K.dot(z.reshape(-1,1))
        
        # Update weight estimates (3.15)
        # If gradient testing is true
        if self.gradient_testing:
            W_next = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
            self.W[0][0,0], self.W[0][1,0] = W_next[0,0], W_next[1,0] 
        else:
            # If gradient testing is false (NORMAL case)
            self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
            self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)

        # Update error covariance matrix (3.16)
        I_minus_K_dot_H = np.identity(K.shape[0]) - K.dot(H)
        self.P = I_minus_K_dot_H.dot(self.P).dot(I_minus_K_dot_H.T) + np.dot(K, (self.R.dot(K.T)))
        #======Update Stage===========#

####

    def _sgd(self, u, y, h, l, step):
        e = h - y
    
        if self.gradient_testing:
            self.W[1] - step*np.hstack((np.outer(e, l), e[:, np.newaxis]))
        else:
            self.W[1] = self.W[1] - step*np.hstack((np.outer(e, l), e[:, np.newaxis]))


        D = (e.dot(self.W[1][:, :-1])*self.dsig(l)).flatten()
        
        if self.gradient_testing:
            W_next = self.W[0] - step*np.hstack((np.outer(D, u), D[:, np.newaxis]))
            self.W[0][0,0], self.W[0][1,0] = W_next[0,0], W_next[1,0]
        else:
            self.W[0] = self.W[0] - step*np.hstack((np.outer(D, u), D[:, np.newaxis]))

#%%
# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

stdev = 0.00055
U = np.arange(-20, 20, 0.2)

sig = lambda V: (1 + np.exp(-V))**-1
Y =  (sig(U * 0.4) + 0.1) + (sig(U * -0.2) + 0.4)
# Y = np.exp(-U**2) + 0.5*np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))

# Create two identical KNN's that will be trained differently
# knn_ekf = KNN(nu=1, ny=1, nl=1, neuron='relu')
hidden_layer = 2
# knn_sif = KNN(nu=1, ny=1, nl=hidden_layer, neuron='logistic')
knn_ekf = KNN(nu=1, ny=1, nl=hidden_layer, neuron='logistic')
knn_sgd = KNN(nu=1, ny=1, nl=hidden_layer, neuron='logistic')
nepochs = 10

print("----Neural Network Structure----")
print(f"Input --> {hidden_layer}neurons --> 1 neuron --> Output")
print(f"epochs: {nepochs}")
print("batch_size: 1")
print("")

# Train
# knn_sif.train(nepochs=nepochs, U=U, Y=Y, method='sif', P=0.5, Q=0.1, R=0.1, pulse_T=0.75)
knn_ekf.train(nepochs=nepochs, U=U, Y=Y, method='ekf', P=0.5, Q=0, R=0.1, pulse_T=0.75)
# knn_sgd.train(nepochs=nepochs, U=U, Y=Y, method='sgd', step=0.05, pulse_T=0.5)

# %%
# results = pd.DataFrame([], columns=["SIF", "SGD", "EKF"])
# for i in range(100):
#     rmse= test_models()
#     results = results.append(rmse, ignore_index=True)

# %%
