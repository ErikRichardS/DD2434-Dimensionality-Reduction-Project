import numpy as np
import sys
import matplotlib.pyplot as plt 

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
    
    d, N = matrix.shape # (209404, 19997) for article dataset, (2500, 1500) for img.
    # Assume columns = data points, rows = dims in matrix. 
    # Flip for now to extract datapoints as rows.
    
    TwoDimInput=False
    
    matrix = matrix.transpose()
    dct_output = []
    
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
        
        t1 = time()
        for item in matrix:
            dct_output_row = DCT(item.transpose(), N, d, k, C, C_inv, TwoDimInput)
            dct_output.append(dct_output_row)
        t2 = time()
        print("Done!")
        
        print("Current iteration time: ", t2-t1)
        
        # Convert list to numpy array, reshape and transpose back
        dct_output = np.array(dct_output)
        dct_output = np.reshape(dct_output, (N, k))
        dct_output = dct_output.transpose()
        
        print("Shape check! Original: (", d, ",", N, "). ", "Current:", dct_output.shape)
        
        error_avr, error_max, error_min = check_quality(matrix, dct_output, N, d, k)
        
        results_avr[i] = error_avr
        results_min[i] = error_min
        results_max[i] = error_max
        results_tim[i] = t2-t1
    
    t3 = time()
    
    print("Total time: ", t3-t1)
    
    return results_avr, results_max, results_min, results_tim


def print_results(res):
    for r in res:
        s = str(r).replace(".", ",")
        print(s)




#indx = [2**i for i in range(1, 11)]
indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]

# matrix = readTextFiles()
matrix = readImageFiles()

# PCA

# res_pca = test_pca(matrix, indx)

# print("Average")
# print_results(res_pca[0])

# print("Max")
# print_results(res_pca[1])

# print("Min")
# print_results(res_pca[2])

# print("Time")
# print_results(res_pca[3])

# print("Percentage")
# print_results(res_pca[4])

# DCT

res_dct_avr, res_dct_max, res_dct_min, res_dct_tim = test_dct(matrix, indx)

print("Average")
print_results(res_dct_avr)

print("Max")
print_results(res_dct_max)

print("Min")
print_results(res_dct_min)

print("Time")
print_results(res_dct_tim)

# RP

"""
res_rp = test_random_projection(matrix, indx)
print("Average")
print(res_dct_avr)
print()
print("Max")
print(res_rp[1])
"""
#

#print(res_pca[0])
#print(res_pca[1])
#print(res_pca[-1])
