import numpy as np
import os
import pca
import matplotlib.pyplot as plt

def compress_images(DATA, k):
    output_dir = 'Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    DATA = np.transpose(DATA)
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    PCS, L = pca.find_pcs(COV)
    PCS = PCS.real
    L = L.real
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    U_t = np.transpose(PCS)
    X_compressed = np.matmul(Z_star, U_t[0:k])
    
    # rescale
    avg = np.average(DATA, axis=0)
    X_compressed = np.array(X_compressed + avg)
    X_compressed *= (255.0 / (X_compressed.max() - X_compressed.min()))

    count = 0
    for x in X_compressed:
        x = x.reshape(60,48)
        s = "image" + str(count) + ".jpg"
        plt.imsave(os.path.join(output_dir, s), x,  cmap='gray', vmin=0, vmax=255)
        count +=1


def load_data(input_dir):
    # Uses os.walk to get all of the image file names and plt.imread to extract the data and then flatten the data
    data = [[plt.imread(root + file_name).flatten() for file_name in files] for root, dirs, files in os.walk(top=input_dir)]

    # Transposes the data so each image's data is a col and each row corresponds to one location
    data = np.transpose(data[0])
    data = np.array(data, dtype=float)
    return data
