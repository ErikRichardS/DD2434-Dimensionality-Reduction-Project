import torch
import numpy as np
import random

# parameter that controls the quality of distance preservation
epsilon = 0.5

# beta-value
# beta = 0.1

# k-value
# k = k_rp(epsilon, n) ?
# k = k_srp(epsilon, n, beta) ?

# number of dimensions in input matrix
# d = matrix.shape[0]
# number of samples in input matrix
# n = matrix.shape[1]

def k_rp(epsilon, n):
  k = 4*np.log(n)/(epsilon**2/2 - epsilon**3/3)
  return k

def k_srp(epsilon, n, beta):
  k = (4 + 2*beta)*np.log(n)/(epsilon**2/2 - epsilon**3/3)
  return k

def center_matrix(matrix):
	n = matrix.shape[1]

	matrix -= ( matrix @ np.ones((n,n)) ) / n

	return matrix

# Singular Value Decomposition
def svd(matrix):
	return np.linalg.svd(matrix)

# Principal Component Analysis
def PCA(matrix, k=2):
	mat = center_matrix(matrix)

	u, s, vh = svd(mat)
	uk = np.transpose(u[:,:k] )
	pca_matrix = uk @ mat

	#TODO: Calculate and return explaination percentage as second variable

	return pca_matrix

# Random Projection
def RandomProjection(matrix, k=2):
	d, n = matrix.shape
	for i in range(0,n):
		# create a matrix with random normal (Gaussian) distributed elements
	    	mean, standard_deviation = 0, 1 
    		r = np.random.normal(mean, standard_deviation, (k, d))
    		# normalize all column vectors in r to unit vectors
    		r_unitnorm = r/np.linalg.norm(r, ord=2, axis=0, keepdims=True)

    		# compute the lower-dimensional random projection matrix
    		rp_matrix = r_unitnorm @ matrix

		# check if it satisfies the desired JL property
		result = check_property(matrix, rp_matrix, n, d, k)
		# if so, return random projection matrix
		if result == True:
			return rp_matrix

def check_property(matrix, rp_matrix, n, d, k):
  	for i in range(0,n):
    		for j in range(0,n):
      			xi = matrix[:,i]
      			xj = matrix[:,j]
      			dist_x = np.linalg.norm(xi-xj)
			
      			f_xi = rp_matrix[:,i]
      			f_xj = rp_matrix[:,j]
      			dist_fx = np.linalg.norm(f_xi-f_xj)
			#dist_fx = np.sqrt(d/k)*np.linalg.norm(f_xi-f_xj)

      			if (abs(dist_x-dist_fx)/dist_x > epsilon):
        			return False
  	return True


# Sparse Random Projection
def SparseRandomProjection(matrix, k=2):
	d = matrix.shape[0]
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



