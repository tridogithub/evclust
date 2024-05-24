# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024

"""
    This module contains functions to run Fuzzy C-means algorithm specified in the paper:
    FUZZY CLUSTERING WITH WEIGHTING OF DATA VARIABLES - Keller 2000 et al.
    (doi.org/10.1142/S0218488500000538)
"""

# ---------Import---------
import numpy as np
from scipy.cluster.vq import kmeans


def calculate_objective_func(x, v, m, w, alpha, beta):
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
        weights = np.tile(w[j, :], (n, 1)) ** alpha
        d2w[:, j] = np.nansum(((x - np.tile(v[j, :], (n, 1))) ** 2) * weights, axis=1)

    tmp1 = (m ** beta) * d2w
    j = np.sum(tmp1)
    return j


def fcm(x, c, w0=None, beta=2, alpha=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """
    Weighted Fuzzy C-means
    Args:
        x: data set
        c: number of clusters
        w: initial weights, If None, the algorithm randomly generate
        beta: the fuzzifier pf membership value
        alpha: the influence of weights
        epsilon: the stopping condition
        init: the initialization of centers
        stop_factor: "weight" for weight change, "center" for center change, None for objective function change
        verbose: display debugging messages

    Returns: A fuzzy partition matrix indicating membership values of objects to each cluster

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
    if w0 is not None:
        if w0.shape[0] != n or w0.shape[1] != d:
            raise ValueError("Invalid size of initial weight matrix.")
    else:
        w0 = np.random.dirichlet(np.ones(d), c)
    if verbose:
        print(f"Initial weights: {w0}")

    j_old = np.inf
    finis = False
    v = None
    m = None
    w = None
    iter = 0
    while not finis:
        # Distance to centers (n x c)
        d2 = np.zeros((n, c))
        for j in range(c):
            d2[:, j] = np.nansum((x - np.tile(v0[j, :], (n, 1))) ** 2, axis=1)

        # weighted distance to centers (n x c)
        d2w = np.zeros((n, c))
        for j in range(c):
            weights = np.tile(w0[j, :], (n, 1)) ** alpha
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

        # Calculating new weights
        w = np.zeros((c, d))
        for j in range(c):
            tmp1 = np.tile(v[j, :], (n, 1))
            tmp2 = (np.tile(m[:, j].reshape(-1, 1), (1, d)) ** beta) * ((x - tmp1) ** 2)
            tmp3 = np.nansum(tmp2, axis=0) ** (1 / (alpha - 1))

            tmp4 = (1 / np.nansum(tmp2, axis=0)) ** (1 / (alpha - 1))
            tmp5 = np.nansum(tmp4)

            w[j, :] = 1 / (tmp3 * np.tile(tmp5, (1, d)))

        j = calculate_objective_func(x, v, m, w, alpha, beta)

        j_change = np.abs(j - j_old)
        v_change = np.linalg.norm(v - v0)
        w_change = np.linalg.norm(w - w0)
        if stop_factor == "weight":
            finis = w_change < epsilon
        elif stop_factor == "center":
            finis = v_change < epsilon
        else:
            finis = j_change < epsilon

        v0 = v
        w0 = w
        j_old = j
        iter += 1
        if verbose:
            print(f"[{iter}, {j}]")

    return {'fuzzy_part': m, 'centers': v, 'weights': w, 'obj_func': j}
