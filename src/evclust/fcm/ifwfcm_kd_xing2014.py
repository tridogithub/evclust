# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024

"""
    This module contains functions to run weighted Fuzzy C-means algorithm specified in the paper:
    Further improvements in Feature-Weighted Fuzzy C-Means - Xing 2014 et al.
    (doi.org/10.1016/j.ins.2014.01.033)
"""

# ---------Import---------
import numpy as np
from scipy.cluster.vq import kmeans


def calculate_objective_func(x, v, m, w, beta):
    """
    Calculate the objective function value
    Args:
        w: weight matrix
        m: membership matrix
        v: centers
        x: data
        beta: the fuzzifier pf membership value
    """
    # weighted distance to centers (n x d)
    n = x.shape[0]
    c = v.shape[0]
    # Calculate kernel function's values matrix
    variance = np.var(x.flatten())  # square of standard deviation
    k = np.zeros((n, c))
    for j in range(c):
        vjw = np.tile(v[j, :] * w, (n, 1))
        xw = x * np.tile(w, (n, 1))
        tmp1 = - np.nansum((xw - vjw) ** 2, axis=1) / (2 * variance)
        k[:, j] = np.exp(tmp1)

    # weighted distance to centers (n x c)
    d2w = 2 - 2 * k

    tmp1 = (m ** beta) * d2w
    j = np.sum(tmp1)
    return j


def fcm(x, c, beta=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """
    Weighted Fuzzy C-means
    Args:
        x: data set
        c: number of clusters
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

    # Initialize weight matrix, all weights have the same value 1/d (d x 1)
    w0 = np.ones(d) / d
    if verbose:
        print(f"Initial weights: {w0}")

    j_old = np.inf
    finis = False
    v = None
    m = None
    w = None
    iteration = 0
    while not finis:
        # Calculate kernel function's values matrix
        variance = np.var(x.flatten())  # square of standard deviation
        k = np.zeros((n, c))
        for j in range(c):
            vjw = np.tile(v0[j, :] * w0, (n, 1))
            xw = x * np.tile(w0, (n, 1))
            tmp1 = - np.nansum((xw - vjw) ** 2, axis=1) / (2 * variance)
            k[:, j] = np.exp(tmp1)

        # weighted distance to centers (n x c)
        d2w = 1 - 2 * k + 1

        # Calculate the next membership matrix
        tmp = 1 / (d2w ** (1 / (beta - 1)))
        tmp2 = np.nansum(tmp, axis=1)
        tmp3 = np.tile(tmp2.reshape(-1, 1), c)
        tmp4 = (d2w ** (1 / (beta - 1))) * tmp3
        m = 1 / tmp4

        # Calculating new centers
        v = np.zeros((c, d))
        for p in range(d):
            tmp1 = np.nansum((m ** beta) * k, axis=0)
            tmp2 = np.tile(x[:, p].reshape(-1, 1), c)
            tmp3 = np.nansum((m ** beta) * k * tmp2, axis=0)
            tmp4 = tmp3 / tmp1
            v[:, p] = tmp4

        # Calculating new weights
        w = np.zeros(d)
        for p in range(d):
            xp = np.tile(x[:, p].reshape(-1, 1), c)
            vp = np.tile(v[:, p], (n, 1))
            w[p] = np.nansum(m * k * ((xp - vp) ** 2))
        tmp1 = np.nansum(1 / w)
        w = 1 / (w * tmp1)

        j = calculate_objective_func(x, v, m, w, beta)

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
        iteration += 1
        if verbose:
            print(f"[{iteration}, {j}]")

    return {'fuzzy_part': m, 'centers': v, 'weights': w, 'obj_func': j}
