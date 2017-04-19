from data.database.helpers.image_database_helper import fetch_all_image_vector_pairs
from sklearn.decomposition import PCA
import numpy as np

from data.database.helpers.pca_database_helper import store_pca_vector_to_db
from helpers.list_helpers import print_progress


def convert_and_store():
	all_image_imgvec_pairs = fetch_all_image_vector_pairs()
	image_filenames = [x[0] for x in all_image_imgvec_pairs]
	image_vectors = np.asarray([x[1] for x in all_image_imgvec_pairs])
	image_count = len(image_filenames)
	del x
	del all_image_imgvec_pairs
	pca = PCA(n_components=10)
	pca_vectors = pca.fit_transform(image_vectors)
	for i in range(image_count):
		store_pca_vector_to_db(image_filenames[i], pca_vectors[i])
		print_progress(i, image_count, prefix="Storing PCA vectors")

if __name__ == '__main__':
	convert_and_store()
