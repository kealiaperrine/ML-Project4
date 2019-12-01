import numpy as np

def compute_Z(X, centering=True, scaling=False):

	Z = X

	if centering:
		mean = np.mean(Z, axis = 0)
		Z = X-mean

	if scaling:
		standard_dev = np.std(Z, axis = 0)
		Z = Z/standard_dev

	return Z

def compute_covariance_matrix(Z):
    Z_t = np.transpose(Z)
    cov = np.matmul(Z_t, Z)
    return cov

def find_pcs(COV):
    w, v = np.linalg.eig(np.array(COV))
    index = w.argsort()[::-1]
    L = np.array(w[index])
    PCS = np.array(v[:, index])
    return PCS, L

# assume k and var are never zero at same time
def project_data(Z, PCS, L, k, var):

	variance = 0

	if k == 0:

		while (variance < var):
			k += 1
			variance = sum(L[0:k])/sum(L)

	Z_star = np.dot(Z, PCS[:,0:k])

	return Z_star