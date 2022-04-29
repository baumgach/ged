import numpy as np


def pairwise_jaccard_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """
    Input parameters A & B need to be nd.arrays of type 'int' or 'bool'
    Returns a matrix of pairwise jaccard distances of shape [N,M]
    """

    assert A.dtype in ["int", "bool"] and B.dtype in ["int", "bool"]
    assert np.all([int(u) in [0, 1] for u in np.unique(A)])
    assert np.all([int(u) in [0, 1] for u in np.unique(B)])

    B = B.transpose((3, 1, 2, 0))
    intersection = np.sum(A & B, axis=(1, 2))
    union = np.sum(A, axis=(1, 2)) + np.sum(B, axis=(1, 2)) - intersection
    pairwise_jaccard_distances = 1 - (intersection / union)

    # Get rid of the potential nan values again
    pairwise_jaccard_distances[(union == 0) & (intersection == 0)] = 1
    pairwise_jaccard_distances[(union == 0) & (intersection > 0)] = 0

    return pairwise_jaccard_distances


def pairwise_L2_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Returns a matrix of pairwise L2 distances of shape [N,M]"""
    all_differences = A - B.transpose((3, 2, 1, 0))
    pairwise_l2_norms = np.linalg.norm(all_differences, axis=1)
    return pairwise_l2_norms


def generalised_energy_distance(
    x: np.ndarray, y: np.ndarray, metric=pairwise_jaccard_distance
) -> float:
    """
    Calculate the (generalised) energy distance (https://en.wikipedia.org/wiki/Energy_distance)
    where x,y are np.ndarrays containing samples of the distributions to be
    compared for a given metric.

        Parameters:
            x (np.ndarray of shape N x Sx x Sy): One set of N samples
            y (np.ndarray of shape M x Sx x Sy): Another set of M samples
            metric (function): a function implementing the desired metric

        Returns:
            The generalised energy distance of the two samples (float)
    """

    assert x.ndim == 3 and y.ndim == 3

    def expectation_of_difference(a, b):
        N, M = a.shape[0], b.shape[0]
        A = np.tile(a[:, :, :, np.newaxis], (1, 1, 1, M))  # N x Sx x Sy x M
        B = np.tile(b[:, :, :, np.newaxis], (1, 1, 1, N))  # M x Sx x Sy x N
        return metric(A, B).mean()

    Exy = expectation_of_difference(x, y)
    Exx = expectation_of_difference(x, x)
    Eyy = expectation_of_difference(y, y)

    ed = np.sqrt(2 * Exy - Exx - Eyy)
    return ed


if __name__ == "__main__":

    N = 1000

    print("Sanity check for L2 distance (i.e. normal energy distance)")
    a = np.random.multivariate_normal(np.zeros((16,)), np.eye(16), size=N).reshape((-1, 4, 4))
    b = np.random.multivariate_normal(np.zeros((16,)), np.eye(16), size=N).reshape((-1, 4, 4))
    c = np.random.multivariate_normal(np.ones((16,)), np.eye(16), size=N).reshape((-1, 4, 4))
    d = np.random.multivariate_normal(np.zeros((16,)), 2 * np.eye(16), size=N * 2).reshape((-1, 4, 4))

    print(generalised_energy_distance(a, a, metric=pairwise_L2_distance))
    print(generalised_energy_distance(a, b, metric=pairwise_L2_distance))
    print(generalised_energy_distance(a, c, metric=pairwise_L2_distance))
    print(generalised_energy_distance(a, d, metric=pairwise_L2_distance))

    print("Sanity check for generalised energy distance with jaccard distance metric")
    a = np.random.randint(2, size=(N, 16)).reshape((-1, 4, 4))
    b = np.random.binomial(1, p=0.5, size=(N, 16)).reshape((-1, 4, 4))
    c = np.random.binomial(1, p=0.1, size=(N, 16)).reshape((-1, 4, 4))
    d = np.random.binomial(1, p=0.0, size=(N * 2, 16)).reshape((-1, 4, 4))

    print(generalised_energy_distance(a, a, metric=pairwise_jaccard_distance))
    print(generalised_energy_distance(a, b, metric=pairwise_jaccard_distance))
    print(generalised_energy_distance(a, c, metric=pairwise_jaccard_distance))
    print(generalised_energy_distance(a, d, metric=pairwise_jaccard_distance))
