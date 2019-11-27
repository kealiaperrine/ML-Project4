import numpy as np

def compute_Z(X, centering=True, scaling=False):
    if centering:
        mean = np.mean(X, axis=0)
        Z = np.array(X-mean)
    if scaling:
        standard_dev = np.std(X, axis = 0)
        Z = np.array(X)
        Z = Z.astype('float')
        for i in range(X.shape[1]):
            Z[:, i] = np.true_divide(X[:,i], standard_dev[i])
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
        variance = var
    else:
        variance = k

    eigenvectors = []
    eigenvalues = []
    for i in range(variance):
        eigenvectors.append(PCS[:,i])
        eigenvalues.append(L[i])
    Z_star =

def main():
    # X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # X = np.array([[0, 1], [-2, -3]])
    X = np.array([[1, 2], [ 2, 6], [3, 4], [5, 7]])
    Z = compute_Z(X, False, True)
    cov = compute_covariance_matrix(Z)
    PCS, L = find_pcs(cov)
    project_data(Z, PCS, L, 1, 0)


if __name__ == "__main__":
    main()