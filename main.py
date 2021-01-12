import numpy as np

from reduction import *
from Parser import *


def test_pca(matrix, dims):
	results = np.zeros(len(dims))
	d, N = matrix.shape


	for i, k in enumerate(dims):
		pca_matrix, explain_percentage = PCA(matrix, k=k)
		pca_matrix = pca_matrix
		error = check_quality(matrix, pca_matrix, N, d, k, scaling=False)

		results[i] = error


	return results


def test_random_projection(matrix, dims):
	return 0


def test_sparse_random_projection(matrix, dims):
	return 0

def test_dct(matrix, dims):
	return 0



matrix = readTextFiles()



res = test_pca(matrix, [2, 4, 8, 16, 32, 64, 128])

print(res)

