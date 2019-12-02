import os
import pca
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compress_images(DATA, k):
    output_dir = 'Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    PCS, L = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    U_t = np.transpose(PCS)
    X_compressed = np.matmul(Z_star, U_t[0:k])

    # rescale
    X_compressed *= (255.0 / X_compressed.max())
    resized = np.array([[0 if i < 0 else i for i in x] for x in X_compressed])

    #s = "image" + str(count) + ".jpg"
    plt.imsave(os.path.join(output_dir, "image.png"), resized,  cmap=cm.gray)


def load_data(input_dir):
    # Uses os.walk to get all of the image file names and plt.imread to extract the data and then flatten the data
    data = [[plt.imread(root + file_name).flatten() for file_name in files] for root, dirs, files in os.walk(top=input_dir)]

    # Transposes the data so each image's data is a row and each column corresponds to one location
    data = np.transpose(data[0])
    data = np.array(data, dtype=float)
    return data
