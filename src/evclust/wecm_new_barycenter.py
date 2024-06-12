# -*- coding: utf-8 -*-
# Van Tri DO (van_tri.do@etu.uca.fr) - France, 2024
"""
This module contains the main function for w-ecm (Weighted ECM clustering) with new equation of barycenters
"""

# ---------------------- Packages------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from scipy.cluster.vq import kmeans


def __get_objective_func_value(w, m, v, F, x, alpha, beta, delta):
    n = m.shape[0]
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)
    d = w.shape[1]

    # weights of singleton sets
    w0k = w[card == 1]  # (c x d)
    w0k2 = w0k ** 2

    vplus = np.zeros((f - 1, d))
    # Calculate (2^c - 1) centroids vplus
    for i in range(1, f):
        fi = F[i, :]
        truc = np.tile(fi, (d, 1)).T
        vplus[i - 1, :] = np.sum(v * w0k2 * truc, axis=0) / np.sum(w0k2 * truc, axis=0)

    # Calculate weighted distances
    dw2 = np.zeros((n, f - 1))
    for j in range(f - 1):
        wj = np.tile(w[j, :], (n, 1))  # Weight of each dimension wrt cluster j, repeated n times
        dw2[:, j] = np.nansum(((x - np.tile(vplus[j, :], (n, 1))) * wj) ** 2, axis=1)

    # Calculate objective function's new value
    mvide = 1 - np.sum(m, axis=1)

    j1 = np.nansum((m ** beta) * dw2[:, :f - 1] * (np.tile(card[:f - 1] ** alpha, (n, 1))) + (delta ** 2) * np.nansum(
        mvide[:f - 1] ** beta))
    return j1


def wecm(x, c, v0=None, alpha=1, beta=2, delta=10, epsilon=1e-3, stopping_factor=None, init="kmeans", disp=True):
    """
    Evidential C-means clustering with new equation of bary-centers, and feature-weight integration
    Args:
        x:
            Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
        c:
            Number of clusters.
        v0:
            Initial prototypes, matrix of size (c x d). If not provided, the prototypes are initialized according to 'init'.
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
    w0 = np.ones((f - 1, d)) / d  # ((2^c-1) x d)
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
    j_old = np.inf
    J = None
    iteration = 0
    while not finis:
        # weights of singleton sets
        w0k = w0[card == 1]  # (c x d)
        w0k2 = w0k ** 2

        vplus = np.zeros((f - 1, d))
        # Calculate (2^c - 1) centroids vplus
        for i in range(1, f):
            fi = F[i, :]
            truc = np.tile(fi, (d, 1)).T
            vplus[i - 1, :] = np.sum(v0 * w0k2 * truc, axis=0) / np.sum(w0k2 * truc, axis=0)

        # Calculate weighted distances
        dw2 = np.zeros((n, f - 1))
        for j in range(f - 1):
            wj = np.tile(w0[j, :], (n, 1))  # Weight of each dimension wrt cluster j, repeated n times
            dw2[:, j] = np.nansum(((x - np.tile(vplus[j, :], (n, 1)))  * wj) ** 2, axis=1)

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
        w = np.zeros((f - 1, d))
        for j in range(f - 1):
            tmp1 = np.tile(vplus[j, :], (n, 1))
            aj = np.tile(card[j], (n, d))
            tmp2 = aj * (np.tile(m[:, j].reshape(-1, 1), (1, d)) ** beta) * ((x - tmp1) ** 2)
            tmp3 = np.nansum(tmp2, axis=0)

            tmp4 = (1 / np.nansum(tmp2, axis=0))
            tmp5 = np.tile(np.sum(tmp4), (1, d))
            w[j, :] = 1 / (tmp3 * tmp5)
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
                            aj = card[j] ** alpha
                            mj_beta = m[:, j] ** beta
                            singleton_index = np.where(F[j + 1, :] == 1)[0]
                            w2_sum = np.sum(w0k[singleton_index, p] ** 2, axis=0)
                            tmp1 = (w0k[k, p] * w0k[l, p] / w2_sum) ** 2
                            tmp2 = aj * (w0[j, p] ** 2) * tmp1 * np.sum(mj_beta)
                            H[l, k] += tmp2

            B = np.zeros((c, 1))
            for k in range(c):
                truc = np.zeros(c)
                truc[k] = 1
                t = np.tile(truc, (f, 1))
                indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[
                    0]  # indices of all Aj including wl
                indices = indices - 1

                for j in indices:
                    aj = card[j] ** alpha
                    mj_beta = m[:, j] ** beta
                    singleton_index = np.where(F[j + 1, :] == 1)[0]
                    w2_sum = np.sum(w0k[singleton_index, p] ** 2, axis=0)
                    tmp1 = ((w0k[k, p]) ** 2) / w2_sum
                    tmp2 = aj * mj_beta * (w0[j, p] ** 2) * tmp1 * x[:, p]
                    B[k, 0] += np.nansum(tmp2)
            vp = np.linalg.solve(H, B)
            v[:, p] = vp.transpose()
        # print(f"Centers: {v}")

        J = __get_objective_func_value(w, m, v, F, x, alpha, beta, delta)
        iteration += 1
        if disp:
            print([iteration, J])

        weights_change = np.abs(np.linalg.norm(w - w0))
        centers_change = np.abs(np.linalg.norm(v - v0))

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
    clus = extractMass(m, F, g=v, W=w, method="wecm", crit=J,
                       param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus
