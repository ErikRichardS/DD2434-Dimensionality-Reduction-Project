import torch

import numpy as np

import random


def center_matrix(matrix):
	n = matrix.shape[1]

	matrix -= ( matrix @ np.ones((n,n)) ) / n

	return matrix

def svd(matrix):
	return np.linalg.svd(matrix)

# Principal Component Analysis
def PCA(X, k=2):
	mat = center_matrix(X)

	u, s, vh = svd(mat)
	uk = np.transpose(u[:,:k] )

	pca_matrix = uk @ mat

	return pca_matrix


# input: d-dimensional matrix, k 
# output: k-dimensional matrix (k << d)
def RandomProjection(X, k=2):
	# create a matrix with random normal (Gaussian) distributed elements
	d = len(X)
	mean, standard_deviation = 0, 1 
	R = np.random.normal(mean, standard_deviation, (k, d))
	# normalize all column vectors in R to unit vectors
	R_normalized = R/np.linalg.norm(R, ord=2, axis=0, keepdims=True)
	
	RP = R_normalized @ X
	return RP



def SparseRandomProjection(matrix, k=2):
	d, _ = matrix.shape
	srp = np.zeros((k, d))

	# 1	  with 1/6 propability
	# 0	  with 4/6 propability
	# -1  with 1/6 propability
	rand_val = [1, 0, 0, 0, 0, -1]

	for i in range(k):
		for j in range(d):
			srp[i,j] = random.choice(rand_val)

	srp_matrix = srp @ matrix

	return srp_matrix



def DCT(input_matrix, k=2):

	return input_matrix



