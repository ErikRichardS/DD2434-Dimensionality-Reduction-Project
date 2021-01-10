import numpy as np

from reduction import *
from Parser import *


def test_pca(matrix, dims):
	results = np.zeros(dims)
	d, N = matrix.shape


	for k in range(2,dims+1):
		pca_matrix, explain_percentage = PCA(matrix, k=k)
		pca_matrix = pca_matrix
		error = check_quality(org_matrix, pca_matrix, N, d, k)

		results[k-1] = error


	return results


def test_random_projection(matrix, dims):
	return 0


def test_sparse_random_projection(matrix, dims):
	return 0

def test_dct(matrix, dims):
	return 0



matrix = readTextFiles()



res = test_pca(matrix, 10)

