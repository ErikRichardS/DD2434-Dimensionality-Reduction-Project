import numpy as np
import sys

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
		error = check_quality(matrix, pca_matrix, N, d, k, scaling=False)

		results_prc[i] = explain_percentage
		results_avr[i] = error

	return results_avr, results_prc


def test_random_projection(matrix, dims):
	return 0


def test_sparse_random_projection(matrix, dims):
	return 0

def test_dct(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    
    d, N = matrix.shape # (209404, 19997) for article dataset
    
    # Assume columns = data points, rows = dims in matrix
    for i, k in enumerate(dims):
        dct_output = np.empty((d, k))
        for datapoint in matrix.transpose():
            # TypeError: sparse matrix length is ambiguous; use getnnz() or shape[0]
            dct_row = DCT(datapoint[0], k)  
            dct_output.append(dct_row)
        dct_output = dct_output.transpose() # Transpose back to parser format
        error = check_quality(matrix, dct_output, N, d, k, scaling=False)
        
        results_avr[i] = error
    
    return results_avr

indx = [2**i for i in range(1, 11)]

matrix = readTextFiles()

res_dct = test_dct(matrix, indx)

# res = test_pca(matrix, indx)

# print(res[0])
# print(res[1])
