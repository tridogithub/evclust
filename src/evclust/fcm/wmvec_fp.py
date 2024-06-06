# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024

"""
    This module contains functions to run Evidential C-means algorithm specified in the paper:
    Adaptive weighted multi-view evidential clustering with feature preference - Zhe Liu 2024 et al.
    (doi.org/10.1016/j.knosys.2024.111770)
"""

# ---------Import---------
import numpy as np
from scipy.cluster.vq import kmeans


def wmvec_fp(x, c, views, lamb=1, gamma=1, alpha=2, beta=2, epsilon=1e-3, init="kmeans", stop_factor=None,
             verbose=False):
    """
    (WMVEC-FP) Weighted multi-view evidential clustering with feature preference
    Args:
        x: data objects (n x d)
        c: number of centers
        views: a matrix of binary number specifies views (sets of features) (H x d)
        lamb: lambda constant, influences the distribution of weights across different views
        gamma: gamma constant, fine-tuning the feature weight distribution
        alpha: penalization exponent
        beta: fuzzier exponent
        epsilon: stopping threshold
        init: the initialization of centers. "kmeans" is default, "None" is for random
        stop_factor: "weight" for weight change, "center" for center change, None for objective function change
        verbose: display debugging messages

    Returns:

    """
    H = views.shape[0]
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

    # Initialize feature weights to 1/d (d is number of dimensions/features)
    w_h0 = np.zeros((H, d))
    # Initialize view weights

    j_old = np.inf
    j = None
    v = None  # centers
    m = None  # Mass matrix
    w_h = None  # feature weights of each view
    r_h = None  # view weights
    finis = False
    iteration = 0
    while not finis:

        # Update mass

        # Update view weights

        # Update centers

        # Update feature weights for each view

        j_change = np.abs(j - j_old)
        v_change = np.linalg.norm(v - v0)
        w_change = np.linalg.norm(w_h - w_h0)
        if stop_factor == "weight":
            finis = w_change < epsilon
        elif stop_factor == "center":
            finis = v_change < epsilon
        elif iteration == 100:
            finis = True
        else:
            finis = j_change < epsilon

        v0 = v
        w_h0 = w_h
        j_old = j
        iteration += 1
        if verbose:
            print(f"[{iteration}, {j}]")
    return None
