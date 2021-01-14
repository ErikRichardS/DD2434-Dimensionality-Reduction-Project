import numpy as np
import sys
import time
import matplotlib.pyplot as plt 

from reduction import *
from Parser import *


def test_pca(matrix, dims):
    results_prc = np.zeros(len(dims))
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        pca_matrix, explain_percentage = PCA(matrix, k=k)
        pca_matrix = pca_matrix
        error_avr, error_max, error_min = check_quality(matrix, pca_matrix, N, d, k)

        results_prc[i] = explain_percentage
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min

    return results_avr, results_max, results_min, results_prc


def test_random_projection(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        rp_matrix = RandomProjection(matrix, k=k)
        error_avr, error_max, error_min = check_quality(matrix, rp_matrix, N, d, k)
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min

    return results_avr, results_max, results_min


def test_sparse_random_projection(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        rp_matrix = SparseRandomProjection(matrix, k=k)
        error_avr, error_max, error_min = check_quality(matrix, rp_matrix, N, d, k, scaling=True)
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min

    return results_avr, results_max, results_min

def test_dct(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    
    d, N = matrix.shape # (209404, 19997) for article dataset, (2500, 1500) for img.
    # Assume columns = data points, rows = dims in matrix. 
    # Flip for now to extract datapoints as rows.
    matrix = matrix.transpose()
    
    time1 = time.time()
    
    TwoDimInput=False
    
    for i, k in enumerate(dims):
    
        # Generate DCT matrices
        if TwoDimInput:
            C = Generate_DCT_Matrix(d, N)
            C_inv = Generate_DCT_Matrix(k, N)
        else:
            C = Generate_DCT_Matrix(d, d)
            C_inv = Generate_DCT_Matrix(k, k)
        
        dct_output = []
        
        # print(C.shape, C_inv.shape, matrix.shape)
        
        for item in matrix:
            dct_output_row = DCT(item.transpose(), N, d, k, C, C_inv, TwoDimInput)
            dct_output.append(dct_output_row)
        
        print("Done!")
        
        time2 = time.time()
        
        print("Current iteration time: ", time2-time1)
        
        # Convert list to numpy array, reshape and transpose back
        dct_output = np.array(dct_output)
        dct_output = np.reshape(dct_output, (N, k))
        dct_output = dct_output.transpose()
        
        print("Shape check! Original: (", d, ",", N, "). ", "Current:", dct_output.shape)
        
        error_avr, error_max, error_min = check_quality(matrix, dct_output, N, d, k)
        
        results_avr[i] = error_avr
        results_min[i] = error_min
        results_max[i] = error_max
    
    time3 = time.time()
    
    print("Total time: ", time3-time1)
    
    return results_avr, results_max, results_min

indx = [2**i for i in range(1, 11)]


# matrix = readTextFiles()
matrix = readImageFiles()

res_dct_avr, res_dct_max, res_dct_min = test_dct(matrix, indx)

#res_rp = test_random_projection(matrix, indx)
print("Average")
print(res_dct_avr)
print()
print("Max")
print(res_dct_max)
print()
print("Min")
print(res_dct_min)

#res_pca = test_pca(matrix, indx)

#print(res_pca[0])
#print(res_pca[1])
#print(res_pca[-1])
