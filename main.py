import torch
import torch.nn as nn

import numpy as np

from reduction import *
from Parser import *






def test_pca(matrix, dims):
	results = np.zeros(dims)
	d, N = matrix.shape

	for k in range(1,dims+1):
		pca_matrix, explain_percentage = PCA(matrix, k=k)
		error = check_quality(matrix, pca_matrix, N, d, k)

		results[k-1] = error


	return results




data_matrix = readTextFiles()





