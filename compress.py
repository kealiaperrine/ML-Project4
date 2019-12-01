import numpy as np
import os
import pca.py as pca


def compress_images(DATA,k):

	Z = pca.compute_z(DATA)
	COV = pca.compute_covariance_matrix(Z)
	PCS, L = pca.find_pcs(COV)
	Z_star = pca.project_data(Z, PCS, L, 1, 0)
	U_t = np.transpose(PCS)

	X_compressed = np.matmul(Z_star, PCS[0:k])

def load_data(input_dir):
	
	#Uses os.walk to get all of the image file names and plt.imread to extract the data and then flatten the data
    data = [[plt.imread(root+file_name).flatten() for file_name in files] for root, dirs, files in os.walk(top = input_dir)]
    
    #Transposes the data so each image's data is a row and each column corresponds to one location
    data = np.transpose(data[0])

    return data