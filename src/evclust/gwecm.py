# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024
"""
This module contains the main function for w-ecm (Weighted ECM clustering) with global weights
"""

# ---------------------- Packages------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
import math
from scipy.cluster.vq import kmeans


def __calcualte_barycenters(v, f, d, F):
    # Calculate (2^c - 1) centroids vplus
    vplus = np.zeros((f - 1, d))
    for i in range(1, f):
        fi = F[i, :]
        truc = np.tile(fi, (d, 1)).T
        vplus[i - 1, :] = np.sum(v * truc, axis=0) / np.sum(fi)

    return vplus


def __get_objective_func_value(w, m, vplus, F, x, alpha, beta, delta, gamma):
    n = m.shape[0]
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    # Calculate weighted distances
    dw2 = np.zeros((n, f - 1))
    wp = np.tile(w, (n, 1))  # Weight of each dimension wrt cluster j, repeated n times
    for j in range(f - 1):
        vj = np.tile(vplus[j, :], (n, 1))
        dw2[:, j] = np.nansum((x - vj) ** 2 * wp, axis=1)

    # Calculate objective function's new value
    w_entropy = gamma * np.sum(w * np.log(w))
    mvide = 1 - np.sum(m, axis=1)

    j1 = np.nansum((m ** beta) * dw2[:, :f - 1] * (np.tile(card[:f - 1] ** alpha, (n, 1))) + (delta ** 2) * np.nansum(
        mvide[:f - 1] ** beta)) + w_entropy
    return j1


def gwecm(x, c, v0=None, gamma=1, alpha=1, beta=2, delta=10, epsilon=1e-3, stopping_factor=None, init="kmeans", disp=True):
    """
    Evidential C-means clustering with new equation of bary-centers, and feature-weight integration
    Args:
        x:
            Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
        c:
            Number of clusters.
        v0:
            Initial prototypes, matrix of size (c x d). If not provided, the prototypes are initialized according to 'init'.
        gamma:
            Parameter of feature weight entropy
        alpha:
            Exponent of the cardinality in the cost function.
        beta:
            Exponent of masses in the cost function.
        delta:
            Distance to the empty set.
        epsilon:
            Minimum amount of improvement.
        stopping_factor:
            default: the change of Objective Function smaller than epsilon
            "weight": the change of weights smaller than epsilon
            "center": the change of centers smaller than epsilon
        init:
            Initialization: "kmeans" (default). "None" is for random initializaion
        disp:
            If True (default), intermediate results are displayed.

    Returns: The credal partition

    """
    # ------------ Initialization -------------
    x = np.array(x)
    n = x.shape[0]
    d = x.shape[1]
    delta2 = delta ** 2
    # Create focal sets
    F = makeF(c, "full", None, True)  # (2^c x c)
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    # ------------- Iterations --------------
    # Initialize V and W -> compute M -> compute new_V and new_W for the next interation.
    w0 = np.ones((1, d)) / d  # (1 x d)
    print(f"Initial weights: \n {w0}")

    if v0 is None:
        if init == "kmeans":
            centroids, distortion = kmeans(x, c)
            v0 = centroids
        else:
            v0 = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * d).reshape(c, d)
    else:
        if v0.shape[0] != c or v0.shape[1] != d:
            raise ValueError("Invalid size of Initial prototypes")
    print(f"Initial prototypes: \n {v0}")

    finis = False
    v = None
    w = None
    m = None
    vplus = None
    j_old = np.inf
    J = None
    iteration = 0
    while not finis and iteration < 200:
        if vplus is None:
            vplus = __calcualte_barycenters(v0, f, d, F)

        # Calculate weighted distances
        dw2 = np.zeros((n, f - 1))
        wp = np.tile(w0, (n, 1))  # Weight of each dimension, repeated n times
        for j in range(f - 1):
            dw2[:, j] = np.nansum((x - np.tile(vplus[j, :], (n, 1))) ** 2 * wp, axis=1)

        # Update memberships
        m = np.zeros((n, f - 1))
        for i in range(n):
            vect0 = dw2[i, :]
            for j in range(f - 1):
                vect1 = (np.tile(dw2[i, j], f - 1) / vect0) ** (1 / (beta - 1))
                vect2 = np.tile(card[j] ** (alpha / (beta - 1)), f - 1) / (card ** (alpha / (beta - 1)))
                vect3 = vect1 * vect2
                m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * dw2[i, j] / delta2) ** (1 / (beta - 1)))
                if np.isnan(m[i, j]):
                    m[i, j] = 1  # in case the initial prototypes are training vectors

        # Update weights
        tmp = np.zeros((1, d))
        for p in range(d):
            tmp1 = np.tile(x[:, p].reshape(-1, 1), (1, f - 1))
            tmp2 = np.tile(vplus[:, p], (n, 1))
            aj = np.tile(card, (n, 1)) ** alpha
            tmp3 = aj * (m ** beta) * (tmp1 - tmp2) ** 2
            tmp[0, p] = np.sum(tmp3)
        tmp = math.e ** (- tmp / gamma)
        w = tmp / np.sum(tmp)
        # print(f"Weights: {w}")

        # Update centers
        v = np.zeros((c, d))
        for p in range(d):
            H = np.zeros((c, c))
            for k in range(c):
                for l in range(c):
                    truc = np.zeros(c)
                    truc[[k, l]] = 1
                    t = np.tile(truc, (f, 1))
                    indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[
                        0]  # indices of all Aj including wk and wl
                    indices = indices - 1

                    if len(indices) == 0:
                        H[l, k] = 0
                    else:
                        for j in indices:
                            aj = card[j] ** (alpha - 2)
                            mj_beta = m[:, j] ** beta
                            wp = w0[0, p]
                            H[l, k] += np.sum(mj_beta * aj * wp)

            B = np.zeros((c, 1))
            for k in range(c):
                truc = np.zeros(c)
                truc[k] = 1
                t = np.tile(truc, (f, 1))
                indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[
                    0]  # indices of all Aj including wl
                indices = indices - 1

                wp = w0[0, p]
                tmp1 = np.tile(card[indices] ** (alpha - 1), (n, 1)) * m[:, indices] ** beta * wp
                tmp2 = np.sum(tmp1, axis=1)
                tmp3 = tmp2.reshape(n, 1)
                xp = x[:, p].reshape(n, 1)
                B[k, 0] = np.sum(xp * tmp3, axis=0)
            vp = np.linalg.solve(H, B)
            v[:, p] = vp.transpose()
        vplus = __calcualte_barycenters(v, f, d, F)
        # print(f"Centers: {vplus}")

        J = __get_objective_func_value(w, m, vplus, F, x, alpha, beta, delta, gamma)
        iteration += 1
        if disp:
            print([iteration, J])

        weights_change = np.abs(np.linalg.norm(w) - np.linalg.norm(w0))
        centers_change = np.abs(np.linalg.norm(v) - np.linalg.norm(v0))

        J_change = np.abs(J - j_old)
        if stopping_factor == "weight":
            finis = weights_change <= epsilon
        elif stopping_factor == "center":
            finis = centers_change <= epsilon
        else:
            finis = J_change <= epsilon
        j_old = J

        v0 = v
        w0 = w

    m = np.concatenate((1 - np.sum(m, axis=1).reshape(n, 1), m), axis=1)
    clus = extractMass(m, F, g=v, gplus=vplus, W=w, method="wecm", crit=J,
                       param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus
