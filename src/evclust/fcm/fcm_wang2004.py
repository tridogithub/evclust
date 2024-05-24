# coding: utf-8
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024


import numpy as np
from scipy.cluster.vq import kmeans


def computing_weights_by_evaluation_func(x):
    epsilon = 1e-6
    n = x.shape[0]

    # Calculate distance between data points
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(x[i] - x[j])

    finis = False
    while not finis:

    return None


def calculate_objective_func(x, v, m, w, beta):
    """
    Calculate the objective function value
    Args:
        w: weight matrix
        m: membership matrix
        v: centers
        x: data
        alpha: the influence of weights
        beta: the fuzzifier pf membership value
    """
    # weighted distance to centers (n x d)
    n = x.shape[0]
    c = v.shape[0]
    d2w = np.zeros((n, c))
    for j in range(c):
        weights = np.tile(w[j, :], (n, 1)) ** 2
        d2w[:, j] = np.nansum(((x - np.tile(v[j, :], (n, 1))) ** 2) * weights, axis=1)

    tmp1 = (m ** beta) * d2w
    j = np.sum(tmp1)
    return j


def fcm(x, c, beta=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """

    Args:
        x:
        c:
        beta:
        epsilon:
        init:
        stop_factor:
        verbose:

    Returns:

    """
    if verbose:
        print(f"Dataset includes {x.shape[0]} instances, and {x.shape[1]} features")

    x = np.array(x)
    n = x.shape[0]
    d = x.shape[1]

    # Initialize centers
    if init == "kmeans":
        centroids, distortion = kmeans(x, c)
        v0 = centroids
    else:
        v0 = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * d).reshape(c, d)
    if verbose:
        print(f"Initial centers: {v0}")

    # Initialize weight matrix
    w0 = computing_weights_by_evaluation_func()

    j_old = np.inf
    finis = False
    v = None
    m = None
    iters = 0
    while not finis:
        # Distance to centers (n x c)
        d2 = np.zeros((n, c))
        for j in range(c):
            d2[:, j] = np.nansum((x - np.tile(v0[j, :], (n, 1))) ** 2, axis=1)

        # weighted distance to centers (n x c)
        d2w = np.zeros((n, c))
        for j in range(c):
            weights = np.tile(w0[j, :], (n, 1)) ** 2
            d2w[:, j] = np.nansum(((x - np.tile(v0[j, :], (n, 1))) ** 2) * weights, axis=1)

        # Calculate the next membership matrix
        tmp = 1 / (d2w ** (1 / (beta - 1)))
        tmp2 = np.nansum(tmp, axis=1)
        tmp3 = np.tile(tmp2.reshape(-1, 1), c)
        tmp4 = (d2w ** (1 / (beta - 1))) * tmp3
        m = 1 / tmp4

        # Calculating new centers
        v = np.zeros((c, d))
        for p in range(d):
            tmp1 = np.nansum(m ** beta, axis=0)
            tmp2 = np.tile(x[:, p].reshape(-1, 1), c)
            tmp3 = np.nansum((m ** beta) * tmp2, axis=0)
            tmp4 = tmp3 / tmp1
            v[:, p] = tmp4

        j = calculate_objective_func(x, v, m, w0, beta)

        j_change = np.abs(j - j_old)
        v_change = np.linalg.norm(v - v0)
        if stop_factor == "center":
            finis = v_change < epsilon
        else:
            finis = j_change < epsilon

        v0 = v
        j_old = j
        iters += 1
        if verbose:
            print(f"[{iters}, {j}]")

    return {'fuzzy_part': m, 'centers': v, 'obj_func': j}
