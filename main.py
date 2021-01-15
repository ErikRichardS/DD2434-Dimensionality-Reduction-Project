import numpy as np
import sys
import matplotlib.pyplot as plt 

from reduction import *
from Parser import *

from time import time

# Written by Erik and was foundation for the other test functions
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


# Written by Erik
def print_results(res):
    for r in res:
        s = str(r).replace(".", ",")
        print(s)



# Everything below was written by Erik
indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]

print("Choose data type:\ntext\nimage\nmusic\n>", end="")
data_type = input()
print("Choose projection function:\npca\nrp\nsrp\ndct\n>", end="")
proj_func = input()

matrix = None
t1 = time()
if data_type == "text":
    matrix = readTextFiles()
elif data_type == "image":
    matrix = readImageFiles()
elif data_type == "music":
    matrix = readMusicFiles()
else:
    print("Invalid type")
    sys.exit()
t2 = time()

print("Load data done. Time: %0.3f" % (t2-t1))


function = None
if proj_func == "pca":
    function = test_pca
elif proj_func == "rp":
    function = test_random_projection
elif proj_func == "srp":
    function = test_sparse_random_projection
elif proj_func == "dct":
    function = test_dct
else:
    print("Invalid type")
    sys.exit()

res_avr = 0
res_max = 0
res_min = 0
res_tim = 0
res_per = 0

if proj_func != "pca":
    res_avr, res_max, res_min, res_tim = function(matrix, indx)
else:
    res_avr, res_max, res_min, res_tim, res_per = function(matrix, indx)

print("Average")
print_results(res_avr)

print("Max")
print_results(res_max)

print("Min")
print_results(res_min)

print("Time")
print_results(res_tim)

if data_type == "pca":
    print("Percentage")
    print_results(res_per)