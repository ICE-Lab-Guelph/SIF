# Implement Unscented Kalman filter (UKF) from scratch as a class

import numpy as np
from scipy.linalg import cholesky


class UnscentedKalmanFilter:
    def __init__(self, fx, hx, R, Q, x_init, P_init, alpha, beta, kapa):
        self.fx = fx
        self.hx = hx
        self.R = R
        self.Q = Q
        self.x = x_init
        self.P = P_init
        self.M = x_init.shape[0]    # No. of state variables (M x 1 state vector)

        self.Wm, self.Wc = self.__compute_sigma_weights(alpha, beta, kapa)
        self.alpha = alpha
        self.beta = beta
        self.kapa = kapa

    # Time state prediction (state prediction according to fx, Q)
    def predict(self):
        Wm = self.Wm
        Wc = self.Wc
        fx = self.fx
        x = self.x
        P = self.P
        Q = self.Q

        sigma_points = self.get_sigma_points(Wm, Wc, x, P)
        self.x, self.P, fx_sigma_points = self.unscented_transform(fx, sigma_points, Wm, Wc, Q)

        self.fx_sigma_points = fx_sigma_points    # Save for use in subsequent update() call

    # Measurement update (innovation) according to observed z, H, R
    def update(self, z):
        Wm = self.Wm
        Wc = self.Wc
        hx = self.hx
        x = self.x
        P = self.P
        R = self.R

        # mean and covariance of predicted measurement h(x) from unscented transform
        zhat, S, hx_sigma_points = self.unscented_transform(hx, self.fx_sigma_points, Wm, Wc, R)
        S_inv = np.linalg.inv(S)

        devs_orig = self.fx_sigma_points - self.x
        devs_transformed = hx_sigma_points - zhat

        cross_cov = devs_orig.T @ np.diag(Wc) @ devs_transformed   # Cross correlation between original sigma points and transformed (by h) points

        K = cross_cov @ S_inv   # Kalman gain

        self.x = x + K @(z - zhat)
        self.P = P - K @ S @ K.T

    def __compute_sigma_weights(self, alpha, beta, kapa):
        L = self.M
        lamda = alpha**2 * (L + kapa) - L
        c = L + lamda

        num_elements = 2*L + 1
        Wm = np.full((num_elements), 1/(2*c))
        Wc = np.full((num_elements), 1/(2*c))

        Wm[0] = lamda/c
        Wc[0] = lamda/c + (1 - alpha**2 + beta)

        return Wm, Wc

    # Generate sigma points around a given mean vector x and covariance P
    def get_sigma_points(self, Wm, Wc, x, P):
        L = self.M
        num_elements = 2*L + 1
        lamda = self.alpha**2 * (L + self.kapa) - L

        sigmas = np.zeros((num_elements, L))
        sigmas[0] = x

        A = cholesky((L + lamda) * P)   # Square root of matrix by Cholesky method

        for i in range(1, L+1):
            sigmas[i] = x + A[i-1]

        for i in range(L+1, 2*L+1):
            sigmas[i] = x - A[i-L-1]

        return sigmas

    def unscented_transform(self, fx, sigma_points, Wm, Wc, noise_cov_mat):
        fx_sigmas = [fx(point) for point in sigma_points]
        transformed_sigmas = np.atleast_2d(fx_sigmas)

        mean = transformed_sigmas.T @ Wm

        devs = transformed_sigmas - mean    # Deviations
        cov = devs.T @ np.diag(Wc) @ devs
        cov = cov + noise_cov_mat

        return mean, cov, transformed_sigmas

