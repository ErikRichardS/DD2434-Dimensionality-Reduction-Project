
import numpy as np
import sys
import matplotlib.pyplot as plt 
from scipy.fft import dct, idct

from reduction import *
from Parser import *

from time import time

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
        
        print(matrix.shape)
        
        t1 = time()
        for item in matrix:
            dct_output_row = dct(item, type=2, norm="ortho")
            dct_output_row = dct_output_row[:k]
            dct_output_row = idct(dct_output_row)
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
        
indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]

matrix = readImageFiles()

res_avr = 0
res_max = 0
res_min = 0
res_tim = 0
res_per = 0

res_avr, res_max, res_min, res_tim = test_dct(matrix, indx)

print("Average")
print_results(res_avr)

print("Max")
print_results(res_max)

print("Min")
print_results(res_min)

print("Time")
print_results(res_tim)