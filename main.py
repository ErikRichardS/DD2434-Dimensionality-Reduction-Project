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
    
    # Reduce amount of dims for RAM reasons
    
    matrix_1 = matrix[:][:((N/4)-1)]
    print(N/4, np.floor(N*(1/4))-1)
    sys.exit()
    matrix_2 = matrix[:][:floor(N*(1/4))-1]
    
    # Assume columns = data points, rows = dims in matrix. Flip for now.        
    
    matrix = matrix.transpose()
    dct_output = []
    
    for item in matrix:
        dct_output_row = DCT(item, item.shape[0], d, k=1000, TwoDimInput=True)
        dct_output.append(dct_output_row)
    
    dct_output = np.array(dct_output)
    dct_output = dct_output.transpose()
    
    error = check_quality(matrix, dct_output, N, d, k, scaling=False)
    
    results_avr[0] = error
    
    return results_avr

indx = [2**i for i in range(1, 11)]

matrix = readTextFiles()

res_dct = test_dct(matrix, indx)
print(res_dct)

# res = test_pca(matrix, indx)

# print(res[0])
# print(res[1])
