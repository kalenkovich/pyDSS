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


def unmix_covariances(C0, C1, n_keep=None, threshold=None, return_mixing=False, return_power=False):
    """
    Calculates the R2*N2*R1 part from formula (7) in de Cheveigné and Simon, 2008
    R*, N*, C* are rotation, normalization, and covariance matrices respectively throughout the code.
    :param C0: total covariance
    :param C1: phase-locked covariance
    :param n_keep: number of components to keep
    :param threshold: threshold for eigenvalues
    :param return_power: return phase-locked and total power of components
    :param return_mixing: return the mixing matrix
    :return: unmixing matrix U and additional results depending on return_mixing and return_power
    """
    assert C0.ndim == 2  # rectangular matrix
    assert C0.shape == C0.T.shape  # square matrix
    assert C0.shape == C1.shape  # same shape
    assert np.all(np.isfinite(C0))
    assert np.all(np.isfinite(C1))

    eigvals, R1 = covariance_pca(C0, n_keep=n_keep, threshold=threshold)
    sd2 = np.sqrt(eigvals)
    N2 = np.diag(1 / sd2)
    N2_inv = np.diag(sd2)

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
    sd3 = np.sqrt(np.diag(U.T @ C0 @ U))
    N3 = np.diag(1 / sd3)
    N3_inv = np.diag(sd3)
    U = U @ N3

    return_values = (U, )

    if return_mixing:
        # The mixing matrix
        # U = R1 * N2 * R2 * N3, filtered data is U' * X
        # The mixing matrix should recover X by doing M * U' * X
        # U' = N3' * R2' * N2' * R1'
        # So M = R1_inv' * N2_inv' * R2_inv' * N3_inv'
        # Since R1_inv = R1', R2_inv = R2', and N2 and N3 are diagonal
        # M = R1 * N2_inv * R2 * N3_inv
        M = R1 @ N2_inv @ R2 @ N3_inv
        assert is_pseudoinverse(U, M)
        return_values += (M, )

    if return_power:
        # Calculate phase-locked and total power of components
        phase_locked_power = np.sum(np.power(C0 @ U, 2), axis=0)
        total_power = np.sum(np.power(C1 @ U, 2), axis=0)
        return_values += phase_locked_power, total_power

    return return_values


def is_pseudoinverse(A, B):
    return (
        np.allclose(B @ A.T @ B, B)
    and np.allclose(A.T @ B @ A.T, A.T)
    and np.allclose(B @ A.T, A @ B.T)
    and np.allclose(A.T @ B, B.T @ A)
    )


def allclose_up_to_sign(a, b, component_axis):
    """
    Checks if arrays a and b are close up to the component sign.
    The function can be applied to mixing and unmixing matrices, or data projected into the component space.
    """
    assert a.shape == b.shape

    # Reshape so that each row corresponds to one component. This will help us in the next step.

    def components_in_rows(c):
        return (np.moveaxis(c, source=component_axis, destination=0)
                .reshape(c.shape[component_axis], -1))

    a_ = components_in_rows(a)
    b_ = components_in_rows(b)

    # Find the signs of the value with the largest absolute value for each component

    n_components = a.shape[component_axis]

    def get_signs(c):
        return np.sign(c[np.arange(n_components), np.argmax(np.abs(c), axis=1)])

    a_signs = get_signs(a_)
    b_signs = get_signs(b_)

    # For broadcasting to work we need signs to have the same number of dimensions as a and b

    target_shape = np.ones(len(a.shape), dtype=int)
    target_shape.put(component_axis, n_components)
    a_signs = a_signs.reshape(target_shape)
    b_signs = b_signs.reshape(target_shape)

    # Multiply by signs and compare

    return np.allclose(a * a_signs, b * b_signs)


def dss(data, cov_uncentered=False, cov_unnormalized=False,
        threshold=None, n_keep=None,
        return_mixing=True, return_power=True):
    C0, C1 = calculate_covariances(data, uncentered=cov_uncentered, unnormalized=cov_unnormalized)
    return unmix_covariances(C0, C1,
                             threshold=threshold,
                             return_mixing=return_mixing, return_power=return_power)
