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


def covariance_pca(R, n_keep=None, threshold=-1, descending=True):
    """
    Run PCA on a covariance matrix
    :param R: the covariance matrix
    :param n_keep: keep only the first n_keep principal components
    :param threshold: removes eigenvalues that are smaller than m * threshold where m is the maximum eigenvalue and
                      corresponding eigenvectors.
                      If threshold is None then it is set to the machine precision times max(R.shape).
                      Ignored if n_keep is not None or if threshold == -1.
    :param descending: if True, returns the results in the descending order of eigenvalues
    :return: eigvals - eigenvlaues, eigvecs - PCA rotation matrix
    """
    eigvals, eigvecs = np.linalg.eigh(R)

    if n_keep is not None:
        eigvals = eigvals[-n_keep:]
        eigvecs = eigvecs[:, -n_keep:]

    else:
        if threshold == -1:
            pass
        else:
            if threshold is None:
                threshold = np.finfo(R[0, 0]).eps * max(R.shape)

            abs_threshold = threshold * max(eigvals)
            eigvecs = eigvecs[:, eigvals > abs_threshold]
            eigvals = eigvals[eigvals > abs_threshold]

    # np.linalg.eigh returns eigenvalues in ascending order, let's fix that
    if descending:
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

    return eigvals, eigvecs


def unmix_covariances(C0, C1, n_keep=None, threshold=None):
    """
    Calculates the R2*N2*R1 part from formula (7) in de Cheveigné and Simon, 2008
    R*, N*, C* are rotation, normalization, and covariance matrices respectively throughout the code.
    :param C0:
    :param C1:
    :param n_keep:
    :param threshold:
    :return:
    """
    assert C0.ndim == 2  # rectangular matrix
    assert C0.shape == C0.T.shape  # square matrix
    assert C0.shape == C1.shape  # same shape
    assert np.all(np.isfinite(C0))
    assert np.all(np.isfinite(C1))

    eigvals, R1 = covariance_pca(C0, n_keep=n_keep, threshold=threshold)
    N2 = np.diag(np.sqrt(1 / eigvals))

    # Covariance of whitened and trial-averaged data.
    # If M is trial-averaged data then its PCA-whitened version is M*R1*N2
    # The covariance is then N2'*R1'*M'*M*R1*N2 = N2'*R1'*C1*R1*N2
    C2 = N2.T @ R1.T @ C1 @ R1 @ N2
    eigvals, R2 = covariance_pca(C2, n_keep=n_keep, threshold=threshold)

    # The unmixing matrix
    U = R1 @ N2 @ R2
    # The order is opposite from that in de Cheveigné and Simon, 2008. The reason is that
    # here we assume that the first dimension is time, unlike in the article.

    # Normalize the components
    N3 = np.diag(1 / np.sqrt(np.diag(U.T @ C0 @ U)))
    U = U @ N3

    # Calculate phase-locked and non-phase-locked power of components
    phase_locked_power = np.sum(np.power(C0 @ U, 2), axis=0)
    non_phase_locked_power = np.sum(np.power(C1 @ U, 2), axis=0)

    return U, phase_locked_power, non_phase_locked_power
