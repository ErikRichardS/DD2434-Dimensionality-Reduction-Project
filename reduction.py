import torch

import numpy as np




# Principal Component Analysis
def PCA(input_matrix):

	return input_matrix



def RandomProjection(input_matrix):
	# create a matrix with random normal (Gaussian) distributed elements
	d = len(X)
	mean, standard_deviation = 0, 1 
	R = np.random.normal(mean, standard_deviation, (k, d))
	# normalize all column vectors in R to unit vectors
	R_normalized = R/np.linalg.norm(R, ord=2, axis=0, keepdims=True)
	
	RP = np.matmul(R_normalized, X)
	return RP



def StochasticRandomProjection(input_matrix):

	return input_matrix



def DCT(input_matrix):

	return input_matrix



