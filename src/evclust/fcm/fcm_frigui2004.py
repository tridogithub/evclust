# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024

"""
    This module contains functions to run Fuzzy C-means algorithm specified in the paper:
    Unsupervised learning of prototypes and attribute weights - Frigui 2004 et al.
    (doi.org/10.1016/j.patcog.2003.08.002)
"""

# ---------Import---------
import numpy as np
from scipy.cluster.vq import kmeans


def __calculate_objective_func_scad1(x, v, u, w, delta, m):
    """
    Calculate the objective function value
    Args:
        m: membership opponent (fuzzier)
        w: weight matrix
        u: membership matrix
        v: centers
        x: data
        delta: coefficient chosen such that balance two components of the objective function (c x 1)

    """
    # weighted distance to centers (n x d)
    n = x.shape[0]
    c = v.shape[0]

    # weighted distance to centers (n x c)
    d2w = np.zeros((n, c))
    for j in range(c):
        weights = np.tile(w[j, :], (n, 1))
        d2w[:, j] = np.nansum(((x - np.tile(v[j, :], (n, 1))) ** 2) * weights, axis=1)

    tmp1 = (u ** m) * d2w
    tmp2 = np.sum(w ** 2, axis=1) * delta
    j = np.sum(tmp1) + np.sum(tmp2)
    return j


def __calculate_objective_func_scad2(x, v, u, w, q, m):
    """
    Calculate the objective function value
    Args:
        m: membership opponent (fuzzier)
        w: weight matrix
        u: membership matrix
        v: centers
        x: data
        q: discrimination opponent (>=1)

    """
    # weighted distance to centers (n x d)
    n = x.shape[0]
    c = v.shape[0]

    # weighted distance to centers (n x c)
    d2w = np.zeros((n, c))
    for j in range(c):
        weights = np.tile(w[j, :], (n, 1)) ** q
        d2w[:, j] = np.nansum(((x - np.tile(v[j, :], (n, 1))) ** 2) * weights, axis=1)

    tmp1 = (u ** m) * d2w
    j = np.sum(tmp1)
    return j


def scad1(x, c, m=2, K=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """
    SCAD1 - Weighted Fuzzy C-means
    Args:
        x: data set
        c: number of clusters
        m: membership opponent (fuzzier) (>= 1)
        K: constant to calculate the Delta
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

    # Initialize weight matrix to 1/d (d is number of dimensions/features)
    w0 = np.ones((c, d)) / d
    if verbose:
        print(f"Initial K: {K}")
        print(f"Initial weights: {w0}")

    j_old = np.inf
    finis = False
    v = None  # centers
    u = None  # membership matrix / fuzzy partition
    w = None  # weights
    j = None
    iteration = 0
    while not finis:
        # Distance to centers (n x c)
        d2 = np.zeros((n, c))
        for j in range(c):
            d2[:, j] = np.nansum((x - np.tile(v0[j, :], (n, 1))) ** 2, axis=1)

        # weighted distance to centers (n x c)
        d2w = np.zeros((n, c))
        for j in range(c):
            weights = np.tile(w0[j, :], (n, 1))
            d2w[:, j] = np.nansum(((x - np.tile(v0[j, :], (n, 1))) ** 2) * weights, axis=1)

        # Calculate the next membership matrix
        tmp = 1 / (d2w ** (1 / (m - 1)))
        tmp2 = np.nansum(tmp, axis=1)
        tmp3 = np.tile(tmp2.reshape(-1, 1), c)
        tmp4 = (d2w ** (1 / (m - 1))) * tmp3
        u = 1 / tmp4

        # Calculating new centers
        v = np.zeros((c, d))
        for p in range(d):
            tmp1 = np.nansum(u ** m, axis=0)

            tmp2 = np.tile(x[:, p].reshape(-1, 1), c)
            tmp3 = np.nansum((u ** m) * tmp2, axis=0)
            tmp4 = tmp3 / tmp1
            v[:, p] = tmp4

        # Calculating new weights
        w20 = w0 ** 2
        tmp1 = np.sum(w20, axis=1)  # (c x 1)
        tmp2 = np.sum(K * (u ** m) * d2w, axis=0)  # (c x 1)
        delta = tmp2 / tmp1

        w = np.zeros((c, d))  # (c x d)
        for p in range(d):
            xip = np.tile(x[:, p].reshape(-1, 1), (1, c))
            vjp = np.tile(v[:, p], (n, 1))
            d2ijp = (xip - vjp) ** 2  # (n x c)

            tmp1 = np.sum((u ** m) * ((d2 / d) - d2ijp), axis=0)
            w[:, p] = tmp1 / (2 * delta) + np.ones(c) / d

        j = __calculate_objective_func_scad1(x, v, u, w, delta, m)

        j_change = np.abs(j - j_old)
        v_change = np.linalg.norm(v - v0)
        w_change = np.linalg.norm(w - w0)
        if stop_factor == "weight":
            finis = w_change < epsilon
        elif stop_factor == "center":
            finis = v_change < epsilon
        elif iteration == 100:
            finis = True
        else:
            finis = j_change < epsilon

        v0 = v
        w0 = w
        j_old = j
        iteration += 1
        if verbose:
            print(f"[{iteration}, {j}]")

    return {'fuzzy_part': u, 'centers': v, 'weights': w, 'obj_func': j}


def scad2(x, c, q=2, m=2, epsilon=1e-3, init="kmeans", stop_factor=None, verbose=False):
    """
    SCAD2 - Weighted Fuzzy C-means
    Args:
        q: discrimination exponent (>= 1)
        x: data set
        c: number of clusters
        m: membership opponent (fuzzier) (>= 1)
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

    # Initialize weight matrix to 1/d (d is number of dimensions/features)
    w0 = np.ones((c, d)) / d
    if verbose:
        print(f"Initial weights: {w0}")

    j_old = np.inf
    finis = False
    v = None  # centers
    u = None  # membership matrix / fuzzy partition
    w = None  # weights
    j = None
    iteration = 0
    while not finis:
        # weighted distance to centers (n x c)
        d2w = np.zeros((n, c))
        for j in range(c):
            weights = np.tile(w0[j, :], (n, 1))
            d2w[:, j] = np.nansum(((x - np.tile(v0[j, :], (n, 1))) ** 2) * weights, axis=1)

        # Calculate the next membership matrix
        tmp = 1 / (d2w ** (1 / (m - 1)))
        tmp2 = np.nansum(tmp, axis=1)
        tmp3 = np.tile(tmp2.reshape(-1, 1), c)
        tmp4 = (d2w ** (1 / (m - 1))) * tmp3
        u = 1 / tmp4

        # Calculating new centers
        v = np.zeros((c, d))
        for p in range(d):
            tmp1 = np.nansum(u ** m, axis=0)

            tmp2 = np.tile(x[:, p].reshape(-1, 1), c)
            tmp3 = np.nansum((u ** m) * tmp2, axis=0)
            tmp4 = tmp3 / tmp1
            v[:, p] = tmp4

        # Calculating new weights
        w = np.zeros((c, d))
        for j in range(c):
            tmp1 = np.tile(v[j, :], (n, 1))
            tmp2 = (np.tile(u[:, j].reshape(-1, 1), (1, d)) ** m) * ((x - tmp1) ** 2)
            tmp3 = np.sum(tmp2, axis=0) ** (1 / (q - 1))

            tmp4 = (1 / np.sum(tmp2, axis=0)) ** (1 / (q - 1))
            tmp5 = np.sum(tmp4, axis=0)

            w[j, :] = 1 / (tmp3 * np.tile(tmp5, (1, d)))

        j = __calculate_objective_func_scad2(x, v, u, w, q, m)

        j_change = np.abs(j - j_old)
        v_change = np.linalg.norm(v - v0)
        w_change = np.linalg.norm(w - w0)
        if stop_factor == "weight":
            finis = w_change < epsilon
        elif stop_factor == "center":
            finis = v_change < epsilon
        elif iteration == 100:
            finis = True
        else:
            finis = j_change < epsilon

        v0 = v
        w0 = w
        j_old = j
        iteration += 1
        if verbose:
            print(f"[{iteration}, {j}]")

    return {'fuzzy_part': u, 'centers': v, 'weights': w, 'obj_func': j}
