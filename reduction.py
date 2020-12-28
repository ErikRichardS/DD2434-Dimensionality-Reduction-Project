import torch
import numpy as np
import random

# parameter that controls the quality of distance preservation
epsilon = 0.01

# k-value
# k = K_rp(epsilon, n) ?
# k = K_srp(epsilon, n) ?

def K_rp(epsilon, n):
  k = 4*np.log(n)/(epsilon^2/2 + epsilon^3/3)
  return k

def center_matrix(matrix):
	n = matrix.shape[1]

	matrix -= ( matrix @ np.ones((n,n)) ) / n

	return matrix

# Singular Value Decomposition
def svd(matrix):
	return np.linalg.svd(matrix)

# Principal Component Analysis
def PCA(X, k=2):
	mat = center_matrix(X)

	u, s, vh = svd(mat)
	uk = np.transpose(u[:,:k] )

	pca_matrix = uk @ mat

	return pca_matrix

# Random Projection
def RandomProjection(matrix, k=2):
	# create a matrix with random normal (Gaussian) distributed elements
	d = len(matrix)
	mean, standard_deviation = 0, 1 
	r = np.random.normal(mean, standard_deviation, (k, d))
	# normalize all column vectors in R to unit vectors
	r_unitnorm = R/np.linalg.norm(R, ord=2, axis=0, keepdims=True)
	
	rp_matrix = r_unitnorm @ matrix
	
	return rp_matrix


# Sparse Random Projection
def SparseRandomProjection(matrix, k=2):
	d, _ = matrix.shape
	srp = np.zeros((k, d))

	# 1	with 1/6 propability
	# 0	with 4/6 propability
	# -1	with 1/6 propability
	rand_val = [1, 0, 0, 0, 0, -1]

	for i in range(k):
		for j in range(d):
			srp[i,j] = random.choice(rand_val)

	srp_matrix = srp @ matrix

	return srp_matrix


# Discrete Cosine Transform 
def DCT(input_matrix, k=2):

	return input_matrix



