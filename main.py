import numpy as np

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
		


	return results_average, results_prc


def test_random_projection(matrix, dims):
	return 0


def test_sparse_random_projection(matrix, dims):
	return 0

def test_dct(matrix, dims):
	return 0



matrix = readTextFiles()

indx = [2**i for i in range(1, 11)]

res = test_pca(matrix, indx)

print(res)

