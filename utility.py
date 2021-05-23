import numpy as np
import matplotlib.pyplot as plt
import collections
import logging

# A global list to keep figures (to be saved at the end of the program)
figures_list = []


def add_figure_to_save(fig):
    figures_list.append(fig)


def save_all_figures(dir):
    for i, fig in enumerate(figures_list):
        filename = dir + '/fig_' + str(i) + '.png'
        fig.savefig(filename)
    print('All figures saved to {}'.format(dir))


def setup_logging(output_dir):
    log_filename = output_dir + '/' + 'run_log.log'

    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_filename, 'w+')],
        level=logging.INFO
    )


def plot(x, y, xlabel=None, y_label=None, title=None, new_figure=True, **kwargs):
    if new_figure:
        fig = plt.figure()
        add_figure_to_save(fig)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
    plt.plot(x, y, **kwargs)
    plt.legend(loc='upper right')


def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples=1):
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len, 1))

        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples


def generate_state_series(x_init, F, Q, n_samples):
    """ Generates a series of state variables
        # Arguments
            x_init: Initial value of state variables (Mx1)
            F: Process state transition matrix (MxM)
            Q: Process noise covariance matrix (MxM)
        # Returns
            A numpy array (M x n_samples)
    """
    M = x_init.shape[0]
    x_series = np.zeros(shape=(M, n_samples))
    x_series[:, 0] = x_init[:, 0]

    for i in range(1, n_samples):
        x_series[:, i] = np.matmul(x_series[:, i-1], F)

    return x_series


def generate_measurement_series(x_series, H, R=None, noise_type='gaussian'):
    """ Generates a series of measurements for a series of state variables
            # Arguments
                x_series: Series of state variables (M x n_samples)
                H: Measurement function matrix (DxM)
                R: Measurement noise covariance matrix (DxD)
            # Returns
                A numpy array (D x n_samples)
        """
    n_samples = x_series.shape[1]
    D = H.shape[0]

    z_series = np.matmul(H, x_series)   # D x n_samples

    if R is not None:
        for dim in range(D):
            mean = 0.
            var = R.diagonal()[dim]
            noise = get_noise_series(noise_type, mean, var, n_samples)
            z_series[dim, :] += noise

    return z_series


def get_noise_series(noise_type, mean, var, n_samples):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, var, n_samples)
    elif noise_type == 'uniform':
        a = (3 * var) ** 0.5  # variance = a^2 / 3 for uniform~(-a, +a)
        noise = np.random.uniform(low=-a, high=a, size=n_samples)
    else:
        assert False

    return noise


def predict_series(ann, X_data, series_length, window):
    pred = ann.predict(X_data)
    y_pred_series = np.zeros(series_length)
    y_pred_series[window + 1:] = pred.reshape(len(pred))

    y_self_pred_series = np.zeros(series_length)
    y_self_pred_series[:window] = X_data[0]
    for i in range(window, series_length):
        X_window = y_self_pred_series[i - window:i]
        y = ann.predict(X_window.reshape(1, window))  # Reshape needed
        y_self_pred_series[i] = y

    return y_pred_series, y_self_pred_series