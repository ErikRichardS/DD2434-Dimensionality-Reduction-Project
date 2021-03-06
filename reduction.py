import numpy as np
from math import sqrt
from math import cos
from math import pi
import random
import sys
import scipy.fftpack as scipy 

import scipy.sparse as sp

from sklearn.decomposition import TruncatedSVD


def k_rp(epsilon, N):
    k = 4*np.log(N)/(epsilon**2/2 - epsilon**3/3)
    return k


def k_srp(epsilon, N, beta):
    k = (4 + 2*beta)*np.log(N)/(epsilon**2/2 - epsilon**3/3)
    return k

# Written by Erik
def center_matrix(matrix):
    N = matrix.shape[1]

    matrix -= (matrix @ np.ones((N, N))) / N

    return matrix


# Written by Erik
# Principal Component Analysis
def PCA(matrix, k=2):
    matrix = np.transpose(matrix)
    svd = TruncatedSVD(n_components=k, random_state=42)

    mat = svd.fit_transform(matrix)

    return np.transpose(mat), np.sum(svd.explained_variance_ratio_)


# Random Projection
def RandomProjection(matrix, k):
    d, N = matrix.shape
    # create a Gaussian distributed random matrix
    mean, standard_deviation = 0, 1
    R = np.random.normal(mean, standard_deviation, (k, d))

    # OLD: 
    # normalize all column vectors in R to unit vectors
    #r_unitnorm = r/np.linalg.norm(r, ord=2, axis=0, keepdims=True)

    # multiply the original d-dimensional matrix with the random projetion matrix R
    # the scalar accounts for the impact on pairwise distances of working in the lower dimensional space k
    scalar = 1/np.sqrt(k)
    rp_matrix = scalar * R @ matrix

    return rp_matrix


def check_quality(ddim_matrix, kdim_matrix, N, d, k):
    # calculate the error in the distance between members of a pair of data vectors, averaged over 100 pairs
    average_error = 0
    max_error = 0
    min_error = float('inf')
    pairs = []
    
    for _ in range(0, 100):
        # pick random vector pair, but make sure it hasn't been used before
        while True:
            i, j = random.sample(list(range(0, N)), 2)
            if [i, j] not in pairs:
                pairs.append([i, j])
                break

        # calculate the distance between the datapoints in the d-dim matrix
        xi = ddim_matrix[:, i]
        xj = ddim_matrix[:, j]
        # calculate the distance between the datapoints in the k-dim matrix
        f_xi = kdim_matrix[:, i]
        f_xj = kdim_matrix[:, j]

        dist_x = sp.linalg.norm(xi-xj)
        dist_fx = np.linalg.norm(f_xi-f_xj)

        # calculate the error
        # TO ADD: Handle if dist_x=0. Might not be necessary since we only get a warning, not an error
        error = abs((dist_x**2-dist_fx**2)/dist_x**2)
        if max_error < error:
            max_error = error
        if min_error > error:
            min_error = error
        average_error += error/100

    return average_error, max_error, min_error


def generate_rp_matrix(matrix, k, rp=True):
    d, N = matrix.shape

    if rp == True:
        iterations = N
    else:
        # TO DO: Add the number of iterations that need to be done for SRP
        pass

    min_error = 1
    best_rp_matrix = None
    for _ in range(0, iterations):
        if rp == True:
            # get random projection matrix
            rp_matrix = RandomProjection(matrix, k)
        else:
            # get sparse random projection matrix
            rp_matrix = SparseRandomProjection(matrix, k)

        # check quality of distance preservation
        error = check_quality(matrix, rp_matrix, N, d, k)
        # save best rp_matrix and the associated error
        if error < min_error:
            min_error = error
            best_rp_matrix = rp_matrix

    return best_rp_matrix, min_error

# Written by Erik
# Sparse Random Projection
def SparseRandomProjection(matrix, k=2):
    d = matrix.shape[0]
    srp = np.random.rand(k, d)
    
    srp = -1*(srp < 0.33) + (srp > 0.67)

    scalar = 1/np.sqrt(k)
    srp = scalar * srp

    srp_matrix = srp @ matrix

    return srp_matrix


def Generate_DCT_Matrix(X, Y):

    C = np.zeros((X, Y))  # Initialize discrete cosine transform matrix

    for rownum in range(X):
        for colnum in range(Y):
            if rownum == 0:
                C[rownum][colnum] = sqrt(1/X)
            else:
                C[rownum][colnum] = sqrt(2/X)*cos((2*colnum+1)*pi*rownum/(2*X))
    return C


def DCT(input_data, N, d, k, C, C_inv, TwoDimInput=False):
    
    if TwoDimInput:
        output_data = C@input_data@C.transpose()
        output_data = output_data[:k][:]
        output_data = C_inv@output_data@C_inv.transpose()      
    else:
        # Apply DCT matrix, cut data, apply inverse DCT
        output_data = C@input_data # dd*dN = dN
        for iteration in range(d-k):
            output_data = np.delete(output_data, output_data.argmin()) # dN --> kN
        output_data = C_inv@output_data # kk*kN = kN

    return output_data

def DCT_SCIPY(input_data, N, d, k, C, C_inv):
    
    output_data = scipy.dct(input_data)
    for iteration in range(d-k):
        output_data = np.delete(output_data, output_data.argmin()) # dN --> kN    
    output_data = scipy.idct(output_data)
    
    return output_data