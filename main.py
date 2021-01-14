import numpy as np
import sys
import time
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
    matrix = matrix.transpose()
    
    time1 = time.time()
    
    TwoDimInput=False
    
    matrix = matrix.transpose()
    dct_output = []
    
    for item in matrix:
        t1 = time()
        dct_output_row = DCT(item, item.shape[0], d, k=1000, TwoDimInput=True)
        t2 = time()
        dct_output.append(dct_output_row)
    
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
    
    return results_avr, results_max, results_min, results_tim


def print_results(res):
    for r in res:
        s = str(r).replace(".", ",")
        print(s)




#indx = [2**i for i in range(1, 11)]
indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]

#matrix = readTextFiles()
#matrix = readImageFiles()
matrix = readMusicFiles()

results = test_random_projection(matrix, indx)

print("Average")
print_results(results[0])

print("Max")
print_results(results[1])

print("Min")
print_results(results[2])

print("Time")
print_results(results[3])

#print("Percentage")
#print_results(results[4])

