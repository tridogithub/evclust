# -*- coding: utf-8 -*-
"""
This module contains the main function for fw-ecm
"""

# ---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from scipy.cluster.vq import kmeans


# ---------------------- fw-ecm------------------------------------------------
def get_weight_focal_set(W, F, d):
    # Calculate weights of 2^c-1 subsets (2^c-1, d)
    f = F.shape[0]
    wplus = np.zeros((f - 1, d))
    for i in range(1, f):
        fi = F[i, :]
        truc = np.tile(fi, (d, 1)).T
        wplus[i - 1, :] = np.sum(W * truc, axis=0) / np.sum(fi)
    return wplus


def get_gradient_matrix(W, M, V, F, X, alpha, beta):
    c = W.shape[0]
    d = W.shape[1]
    n = M.shape[0]
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    # Calculate sum of weights in a focal set(except empty set) in each dimension
    sumWk = np.zeros((f - 1, d))  # matrix of size (2^c -1, d)
    for i in range(1, f):
        k_indices = np.where(F[i, :] == 1)[0]
        sumWk[i - 1, :] = np.sum(W[k_indices, :], axis=0)

    gradient_j1 = np.zeros((c, d))
    for k in range(c):
        for p in range(d):
            truc = np.zeros(c)
            truc[k] = 1
            t = np.tile(truc, (f, 1))
            indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]  # indices of all Aj including wl
            indices = indices - 1

            xip = np.tile(X[:, p].reshape(n, 1),
                          (1, indices.size))  # size (n x j), j is number of sets containing \omega_{k}
            vjp = np.tile(V[indices, p], (n, 1))  # size (n x j), j is number of sets containing \omega_{k}
            gradient_j1i = np.tile(card[indices] ** (alpha - 1), (n, 1)) * (M[:, indices] ** beta) * sumWk[k, p] * (
                    (xip - vjp) ** 2)
            gradient_j1[k, p] = np.sum(np.sum(gradient_j1i, axis=1), axis=0)

    return gradient_j1


def get_j1_objective_func_value(W, M, V, F, X, alpha, beta, delta, noise_w):
    n = M.shape[0]
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)
    d = W.shape[1]

    wplus = get_weight_focal_set(W, F, d)

    # calculation of weighted distances to centers (n x (c-1))
    DW = np.zeros((n, f - 1))
    for j in range(f - 1):
        wj = np.tile(wplus[j, :], (n, 1))  # Weight of each dimension wrt cluster j, repeated n times
        DW[:, j] = np.nansum(((X - np.tile(V[j, :], (n, 1))) * wj) ** 2, axis=1)
    # Calculate objective function's new value
    mvide = 1 - np.sum(M, axis=1)

    j1 = np.nansum((M ** beta) * DW[:, :f - 1] * np.tile(card[:f - 1] ** alpha, (n, 1))) + (np.sum(noise_w, axis=0)) * (
                delta ** 2) * np.nansum(mvide[:f - 1] ** beta)
    return j1


def projected_gradient_descent_jw(start_W, M, V, F, X, alpha, beta, delta, noise_w, learning_rate=0.001, iterations=100,
                                  stopping_threshold=1e-3):
    d = start_W.shape[1]
    c = start_W.shape[0]
    # Calculate project matrix P
    e = np.ones((d, 1))
    e_norm2 = np.linalg.norm(e) ** 2
    P = np.identity(d) - (1 / e_norm2) * np.dot(e, e.transpose())

    old_j1 = get_j1_objective_func_value(start_W, M, V, F, X, alpha, beta, delta, noise_w)
    for i in range(iterations):
        W = np.zeros((start_W.shape[0], start_W.shape[1]))
        gradientJ = get_gradient_matrix(start_W, M, V, F, X, alpha, beta)

        # Calculate new weights
        for k in range(c):
            wk = start_W[k, :].reshape((d, 1))
            gradientJk = gradientJ[k, :]
            new_wk = wk - learning_rate * (np.dot(P, gradientJk.reshape(d, 1)))
            W[k, :] = new_wk.transpose()

        new_j1 = get_j1_objective_func_value(W, M, V, F, X, alpha, beta, delta, noise_w)

        # Check stopping condition
        variance = old_j1 - new_j1
        if variance <= stopping_threshold:
            if variance > 0:
                start_W = W
            break
        else:
            start_W = W
            old_j1 = new_j1

    return start_W


def get_noise_cluster_weights(x):
    """
    Args:
        x: dataset, a matrix of size (nxq)

    Returns: a vector of weights of noise cluster

    """
    x_mean = np.mean(x, axis=0)
    x_deviation = np.abs(x - (np.tile(x_mean, (x.shape[0], 1))))
    x_max_deviation = np.max(x_deviation, axis=0)
    r = x_max_deviation / np.sum(x_deviation, axis=0)
    r_sum = np.sum(r, axis=0)
    noise_w = r / r_sum
    return noise_w


def fwecm(x, c, g0=None, W=None, type='full', pairs=None, Omega=True, ntrials=1, alpha=1, beta=2, delta=10,
          epsi=1e-3, init="kmeans", disp=True):
    """
    Evidential c-means algorithm. `ecm` Computes a credal partition from a matrix of attribute data using the Evidential c-means (ECM) algorithm.

    ECM is an evidential version algorithm of the Hard c-Means (HCM) and Fuzzy c-Means (FCM) algorithms.
    As in HCM and FCM, each cluster is represented by a prototype. However, in ECM, some sets of clusters
    are also represented by a prototype, which is defined as the center of mass of the prototypes in each
    individual cluster. The algorithm iteratively optimizes a cost function, with respect to the prototypes
    and to the credal partition. By default, each mass function in the credal partition has 2^c focal sets,
    where c is the supplied number of clusters. We can also limit the number of focal sets to subsets of
    clusters with cardinalities 0, 1 and c (recommended if c>=10), or to all or some selected pairs of clusters.
    If initial prototypes g0 are provided, the number of trials is automatically set to 1.

    Parameters:
    ----------
    x:
        input matrix of size n x d, where n is the number of objects and d is the number of attributes.
    c:
        Number of clusters.
    g0:
        Initial prototypes, matrix of size c x d. If not supplied, the prototypes are initialized randomly.
    W:
        Initial weight matrix which has size c x d. If not supplied, the weight matrix is initialized randomly.
    type:
        Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega;
            "pairs": empty set, singletons, Omega, and all or selected pairs).
    pairs:
        Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
    Omega:
        Logical. If True (default), the whole frame is included (for types 'simple' and 'pairs').
    ntrials (int):
        Number of runs of the optimization algorithm (set to 1 if g0 is supplied).
    alpha (float):
        Exponent of the cardinality in the cost function.
    beta (float):
        Exponent of masses in the cost function.
    delta (float):
        Distance to the empty set.
    epsi (float):
        Minimum amount of improvement.
    init (str):
        Initialization: "kmeans" (default) or "rand" (random).
    disp (bool):
        If True (default), intermediate results are displayed.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    References:
    ----------
    M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
      Pattern Recognition, Vol. 41, Issue 4, pages 1384--1397, 2008.

    Examples:
    --------
    """

    # ---------------------- initialisations --------------------------------------

    x = np.array(x)
    n = x.shape[0]
    d = x.shape[1]
    delta2 = delta ** 2

    # Initialize the weight vector of noise cluster
    noise_w = get_noise_cluster_weights(x)
    print(f"Intitial noise weights: {noise_w}")

    if (ntrials > 1) and (g0 is not None):
        print('WARNING: ntrials>1 and g0 provided. Parameter ntrials set to 1.')
        ntrials = 1

    F = makeF(c, type, pairs, Omega)
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    if W is not None:
        if not (W.shape[0] == c and W.shape[1] == d):
            raise ValueError("Invalid size of inputted weight matrix.")
    else:
        # randomly initialize weight matrix (c x d)
        W = np.random.dirichlet(np.ones(d), c)
        # Calculate weights of focal sets ((2^c -1) x d)
        wplus = get_weight_focal_set(W, F, d)

    # ------------------------ iterations--------------------------------
    Jbest = np.inf
    for itrial in range(ntrials):
        if g0 is None:
            if init == "kmeans":
                centroids, distortion = kmeans(x, c)
                g = centroids
            else:
                g = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * d).reshape(c, d)
        else:
            g = g0
        pasfini = True
        Jold = np.inf
        gplus = np.zeros((f - 1, d))
        iter = 0
        while pasfini:
            iter += 1
            # Calculate (2^c - 1) centroids gplus
            for i in range(1, f):
                fi = F[i, :]
                truc = np.tile(fi, (d, 1)).T
                gplus[i - 1, :] = np.sum(g * truc, axis=0) / np.sum(fi)

            # calculation of distances to centers (n x (c-1))
            D = np.zeros((n, f - 1))
            for j in range(f - 1):
                D[:, j] = np.nansum((x - np.tile(gplus[j, :], (n, 1))) ** 2, axis=1)

            # calculation of weighted distances to centers (n x (c-1))
            DW = np.zeros((n, f - 1))
            for j in range(f - 1):
                wj = np.tile(wplus[j, :], (n, 1))  # Weight of each dimension wrt cluster j, repeated n times
                DW[:, j] = np.nansum(((x - np.tile(gplus[j, :], (n, 1))) * wj) ** 2, axis=1)

            # Calculation of masses
            m = np.zeros((n, f - 1))
            for i in range(n):
                vect0 = DW[i, :]
                for j in range(f - 1):
                    vect1 = (np.tile(DW[i, j], f - 1) / vect0) ** (1 / (beta - 1))
                    vect2 = np.tile(card[j] ** (alpha / (beta - 1)), f - 1) / (card ** (alpha / (beta - 1)))
                    vect3 = vect1 * vect2
                    m[i, j] = 1 / (
                            np.sum(vect3) + (card[j] ** alpha * DW[i, j] / (np.sum(noise_w, axis=0) * delta2)) ** (
                            1 / (beta - 1)))
                    if np.isnan(m[i, j]):
                        m[i, j] = 1  # in case the initial prototypes are training vectors

            # Calculation of centers
            # old_V = g
            V = np.zeros((c, d))
            # Calculation of centers at p-th dimension
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
                            for jj in range(len(indices)):
                                j = indices[jj]
                                mj = m[:, j] ** beta
                                H[l, k] += np.sum(mj) * card[j] ** (alpha - 2) * (wplus[j, p] ** 2)

                # Construction of the B matrix
                B = np.zeros((c, 1))
                for l in range(c):
                    truc = np.zeros(c)
                    truc[l] = 1
                    t = np.tile(truc, (f, 1))
                    indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[
                        0]  # indices of all Aj including wl
                    indices = indices - 1

                    wjp = np.tile(wplus[indices, p] ** 2, (n, 1))
                    mi = np.tile(card[indices] ** (alpha - 1), (n, 1)) * m[:, indices] ** beta * wjp
                    s = np.sum(mi, axis=1)

                    mats = np.tile(s.reshape(n, 1), (1, 1))
                    ximp = x[:, p].reshape(n, 1) * mats
                    B[l, 0] = np.sum(ximp, axis=0)

                Vp = np.linalg.solve(H, B)
                V[:, p] = Vp.transpose()
            g = V

            # Calculate weights
            W = projected_gradient_descent_jw(W, M=m, V=gplus, F=F, X=x, alpha=alpha, beta=beta, delta=delta,
                                              noise_w=noise_w,
                                              learning_rate=0.001, iterations=100,
                                              stopping_threshold=1e-6)
            wplus = get_weight_focal_set(W, F, d)

            # Calculate objective function's new value
            # mvide = 1 - np.sum(m, axis=1)
            # J = np.nansum((m ** beta) * D * np.tile(card.reshape(1, f-1), (n, 1))) + delta2 * np.nansum(mvide ** beta)
            # J = np.nansum((m ** beta) * D[:, :f - 1] * np.tile(card[:f - 1] ** alpha, (n, 1))) + delta2 * np.nansum(
            #     mvide[:f - 1] ** beta)
            J = get_j1_objective_func_value(W, m, gplus, F, x, alpha, beta, delta, noise_w)

            if disp:
                print([iter, J])
            pasfini = (np.abs(J - Jold) > epsi)
            Jold = J

        if J < Jbest:
            Jbest = J
            mbest = m
            gbest = g
            wbest = W
        res = np.array([itrial, J, Jbest])
        res = np.squeeze(res)
        if ntrials > 1:
            print(res)

    m = np.concatenate((1 - np.sum(mbest, axis=1).reshape(n, 1), mbest), axis=1)
    clus = extractMass(m, F, g=gbest, W=wbest, method="ecm", crit=Jbest,
                       param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus
