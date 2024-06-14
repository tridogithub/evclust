# coding: utf-8
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024


import numpy as np
from scipy.cluster.vq import kmeans
import math


# ---------------- (Start) Functions for calculating new weights ---------------------------
def get_new_weights(w0, x, w_beta, learning_rate):
    n = x.shape[0]
    d = w0.shape[0]

    distance = np.zeros((n, n))
    w_distances = np.zeros((n, n))
    pij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(x[i] - x[j])
            w_distances[i, j] = np.linalg.norm(x[i] * w0 - x[j] * w0)
            pij[i, j] = 1 / (1 + w_beta * distance[i, j])

    new_w = np.zeros(d)
    for p in range(d):
        sum_p = 0
        for i in range(n):
            for j in range(i):
                tmp1 = (1 - 2 * pij[i, j]) * (-w_beta * w0[p] * (x[i, p] - x[j, p]) ** 2)
                tmp2 = (w_distances[i, j] * ((1 + w_beta * w_distances[i, j]) ** 2))
                if tmp1 != 0 and tmp2 != 0:
                    sum_p += tmp1 / tmp2
        delta_w_p = - learning_rate * sum_p * 1 / (n * (n - 1))
        new_w[p] = w0[p] + delta_w_p

    return new_w


def calculate_evaluation_func(x, w, w_beta):
    n = x.shape[0]

    # Calculate distance between data points
    distance = np.zeros((n, n))
    w_distances = np.zeros((n, n))
    pij = np.zeros((n, n))
    w_pij = np.zeros((n, n))
    e_func = 0
    for i in range(n):
        for j in range(i):
            distance[i, j] = np.linalg.norm(x[i] - x[j])
            w_distances[i, j] = np.linalg.norm(x[i] * w - x[j] * w)
            pij[i, j] = 1 / (1 + w_beta * distance[i, j])
            w_pij[i, j] = 1 / (1 + w_beta * w_distances[i, j])
            e_func += (1 / 2) * (w_pij[i, j] * (1 - pij[i, j]) - pij[i, j] * (1 - w_pij[i, j]))
    e_func = e_func * (2 / (n * (n - 1)))

    return e_func


def computing_weights_by_minimizing_evaluation_func(x, w_beta):
    epsilon = 1e-3
    learning_rate = 0.3

    d = x.shape[1]

    w0 = np.ones(d) / d
    print(f"Initial weights: {w0}")
    w = np.zeros(d)
    finis = False
    iteration = 0
    while not finis and iteration <= 100:
        e0 = calculate_evaluation_func(x, w0, w_beta)
        w = get_new_weights(w0, x, w_beta, learning_rate)
        e = calculate_evaluation_func(x, w, w_beta)

        e_change = np.abs(e - e0)
        finis = e_change < epsilon
        w0 = w
        iteration += 1
        print(f"[{iteration}, {e}]]")
    print(f"Final initial weights: {w}")
    return w


# ---------------- (END) Functions for calculating new weights ---------------------------

def calculate_objective_func(x, v, m, w, beta):
    """
    Calculate the objective function value
    Args:
        w: weight matrix
        m: membership matrix (n x c)
        v: centers
        x: data
        beta: the fuzzifier pf membership value
    """
    # weighted distance to centers (n x d)
    n = x.shape[0]
    c = v.shape[0]
    d2w = np.zeros((n, c))
    for j in range(c):
        weights = np.tile(w, (n, 1)) ** 2
        d2w[:, j] = np.nansum(((x - np.tile(v[j, :], (n, 1))) ** 2) * weights, axis=1)

    tmp1 = (m ** beta) * d2w
    j = np.sum(tmp1)
    return j


def fcm(x, c, w_beta, beta=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """

    Args:
        x: data set
        c: number of clusters
        w_beta: parameter for initializing weights
        beta: the fuzzifier pf membership value
        epsilon: the stopping condition
        init: the initialization of centers. 'kmeans' - default. 'None' - randomly
        stop_factor: "center" for center change, None for objective function change
        verbose: display debugging messages

    Returns:
        A fuzzy partition matrix indicating membership values of objects to each cluster
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
    w0 = computing_weights_by_minimizing_evaluation_func(x, w_beta)

    j_old = np.inf
    finis = False
    v = None
    m = None
    iters = 0
    j = None
    while not finis:
        # weighted distance to centers (n x c)
        d2w = np.zeros((n, c))
        for j in range(c):
            weights = np.tile(w0, (n, 1)) ** 2
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
