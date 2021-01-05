import numpy as np
import random

# number of dimensions in input matrix
# d = matrix.shape[0]
# number of samples in input matrix
# N = matrix.shape[1]


def k_rp(epsilon, N):
    k = 4*np.log(N)/(epsilon**2/2 - epsilon**3/3)
    return k


def k_srp(epsilon, N, beta):
    k = (4 + 2*beta)*np.log(N)/(epsilon**2/2 - epsilon**3/3)
    return k


def center_matrix(matrix):
    N = matrix.shape[1]

    matrix -= (matrix @ np.ones((N, N))) / N

    return matrix


# Singular Value Decomposition
def svd(matrix):
    return np.linalg.svd(matrix)


# Principal Component Analysis
def PCA(matrix, k=2):
    mat = center_matrix(matrix)

    u, s, vh = svd(mat)
    uk = np.transpose(u[:, :k])
    pca_matrix = uk @ mat

    part = 0
    total = 0
    for i in range(len(s)):
        s2 = s[i]*s[i]
        total += s2
        if i < k:
            part += s2

    return pca_matrix, (part / total)


# Random Projection
def RandomProjection(matrix, k, d, N):
        # create a matrix with random normal (Gaussian) distributed elements
    mean, standard_deviation = 0, 1
    r = np.random.normal(mean, standard_deviation, (k, d))
    # normalize all column vectors in r to unit vectors
    r_unitnorm = r/np.linalg.norm(r, ord=2, axis=0, keepdims=True)

    # compute the lower-dimensional random projection matrix
    rp_matrix = r_unitnorm @ matrix

    return rp_matrix


def check_quality(ddim_matrix, kdim_matrix, N, d, k, scaling=True):
    max_error = 0
    # TO ADD: instead of going through all combinations of points, randomize 100 points
    for i in range(0, N):
        for j in range(0, N):
            xi = ddim_matrix[:, i]
            xj = ddim_matrix[:, j]
            dist_x = np.linalg.norm(xi-xj)

            f_xi = kdim_matrix[:, i]
            f_xj = kdim_matrix[:, j]
            if scaling == True:
                dist_fx = np.sqrt(d/k)*np.linalg.norm(f_xi-f_xj)
            else:
                dist_fx = np.linalg.norm(f_xi-f_xj)

                # TO ADD: Handle if dist_x=0
            error = abs((dist_x**2-dist_fx**2)/dist_x**2)
            if error > max_error:
                max_error = error

    return max_error


def generate_rp_matrix(matrix, k, rp=True):
    d, N = matrix.shape

    if rp == True:
        iterations = N
    else:
        pass

    min_error = 1
    best_rp_matrix = None
    for i in range(0, iterations):
        if rp == True:
            # get random projection matrix
            rp_matrix = RandomProjection(matrix, k, d, N)
        else:
            # get sparse random projection matrix
            rp_matrix = SparseRandomProjection(matrix, k)

        # check quality of distance preservation
        error = check_quality(matrix, rp_matrix, N, d, k, scaling=rp)
        # save best rp_matrix and the associated error
        if error < min_error:
            min_error = error
            best_rp_matrix = rp_matrix

    return best_rp_matrix, min_error


# Sparse Random Projection
def SparseRandomProjection(matrix, k=2):
    d = matrix.shape[0]
    srp = np.zeros((k, d))

    # 1	with 1/6 propability
    # 0	with 4/6 propability
    # -1	with 1/6 propability
    rand_val = [1, 0, 0, 0, 0, -1]

    for i in range(k):
        for j in range(d):
            srp[i, j] = random.choice(rand_val)

    srp_matrix = srp @ matrix

    return srp_matrix


# Discrete Cosine Transform
def DCT(input_matrix, k=2):

    return input_matrix
