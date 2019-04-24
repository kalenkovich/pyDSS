import numpy as np


def calculate_covariances(data, uncentered=False, unnormalized=False):
    """
    Calculates averaged per-trial covariances and covariance of trial-averaged data.
    :param data: n_samples x n_channels x n_trials matrix of data
    :param uncentered: if True, does not subtract the time-axis mean
    :param unnormalized: if True, no covariance are divided by (n_samples - 1) and R0 is a sum rather than a mean of
                         covariance matrices
    :return: R0 - averaged per-trial covariances
             R1 - covariance of trial-averaged data
    """
    if not uncentered:
        data = data - np.mean(data, axis=0)

    R0 = sum([X.T @ X for X in np.moveaxis(data, 2, 0)])  # Iteration is over trials

    M = np.mean(data, axis=2)  # Average over trials
    R1 = M.T @ M

    if not unnormalized:
        n_samples, n_trials = data.shape[:2]
        R0 /= (n_samples - 1) * n_trials
        R1 /= (n_samples - 1)

    return R0, R1
