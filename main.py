import numpy as np
import sys

from reduction import *
from Parser import *

from time import time

def test_pca(matrix, dims):
    results_prc = np.zeros(len(dims))
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    results_tim = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        t1 = time()
        pca_matrix, explain_percentage = PCA(matrix, k=k)
        t2 = time()
        pca_matrix = pca_matrix
        error_avr, error_max, error_min = check_quality(matrix, pca_matrix, N, d, k)

        results_prc[i] = explain_percentage
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min
        results_tim[i] = t2-t1

    return results_avr, results_max, results_min, results_tim, results_prc


def test_random_projection(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    results_tim = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        t1 = time()
        rp_matrix = RandomProjection(matrix, k=k)
        t2 = time()
        error_avr, error_max, error_min = check_quality(matrix, rp_matrix, N, d, k)
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min
        results_tim[i] = t2-t1

    return results_avr, results_max, results_min, results_tim


def test_sparse_random_projection(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    results_tim = np.zeros(len(dims))

    d, N = matrix.shape

    for i, k in enumerate(dims):
        t1 = time()
        rp_matrix = SparseRandomProjection(matrix, k=k)
        t2 = time()
        error_avr, error_max, error_min = check_quality(matrix, rp_matrix, N, d, k)
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min
        results_tim[i] = t2-t1

    return results_avr, results_max, results_min, results_tim

def test_dct(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    results_tim = np.zeros(len(dims))
    
    d, N = matrix.shape # (209404, 19997) for article dataset
    
    # Reduce amount of dims for RAM reasons
    
    #matrix_1 = matrix[:][:((N/4)-1)]
    #print(N/4, np.floor(N*(1/4))-1)
    #matrix_2 = matrix[:][:floor(N*(1/4))-1]
    
    # Assume columns = data points, rows = dims in matrix. Flip for now.        
    
    matrix = matrix.transpose()
    dct_output = []
    
    for item in matrix:
        t1 = time()
        dct_output_row = DCT(item, item.shape[0], d, k=1000, TwoDimInput=True)
        t2 = time()
        dct_output.append(dct_output_row)
    
    dct_output = np.array(dct_output)
    dct_output = dct_output.transpose()
    
    error = check_quality(matrix, dct_output, N, d, k=1000)
    
    results_avr[0] = error
    
    return results_avr, results_max, results_min, results_tim


def print_results(res):
    for r in res:
        s = str(r).replace(".", ",")
        print(s)




#indx = [2**i for i in range(1, 11)]
indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]

matrix = readTextFiles()

res_pca = test_pca(matrix, indx)

print("Average")
print_results(res_pca[0])

print("Max")
print_results(res_pca[1])

print("Min")
print_results(res_pca[2])

print("Time")
print_results(res_pca[3])

print("Percentage")
print_results(res_pca[4])

#matrix = readImageFiles()

#res_dct = test_dct(matrix, indx)

"""
res_rp = test_random_projection(matrix, indx)
print("Average")
print(res_rp[0])
print()
print("Max")
print(res_rp[1])
"""
#

#print(res_pca[0])
#print(res_pca[1])
#print(res_pca[-1])
