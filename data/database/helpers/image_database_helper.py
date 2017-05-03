import os
import sys

import numpy as np

file_par_dir = os.path.join(__file__, os.pardir)
file_par_par_dir = os.path.join(file_par_dir, os.pardir)
file_par_par_par_dir = os.path.join(file_par_par_dir, os.pardir)
ROOT_DIR = os.path.dirname((os.path.abspath(file_par_par_par_dir))) + "/"
sys.path.append(ROOT_DIR)

import sqlite_wrapper as wrapper


def store_image_vector_to_db(image_name, vector):
	wrapper.db_insert_image_vector(image_name, vector)


def fetch_all_image_names():
	return [x[0] for x in wrapper.db_keys_images()]


def fetch_all_image_names_with_class(class_string):
	return [(x[0], x[1]) for x in wrapper.db_filenames_by_class(class_string)]


def fetch_image_vector(image_name):
	return wrapper.db_get_image_vector(image_name)[0]


def fetch_all_image_vector_pairs():
	return wrapper.db_all_filename_img_vec_pairs()


def fetch_filename_from_image_vector(image_vector):
	return wrapper.db_get_filename_from_image_vector(image_vector)


def update_image_vectors(filename_image_vector_tuples):
	return wrapper.db_insert_image_vector_list(filename_image_vector_tuples)


def normalize_abs_image_vectors():
	print("Loading data...")
	tr_im_image_vector_tuples = wrapper.db_all_filename_img_vec_pairs()
	print("Loaded data.")
	for i in range(len(tr_im_image_vector_tuples)):
		filename, vector = tr_im_image_vector_tuples[i]
		tr_im_image_vector_tuples[i] = np.asarray(l2norm(vector)), filename
		print_progress(i, len(tr_im_image_vector_tuples), prefix="Normilizing all images")

	update_image_vectors(tr_im_image_vector_tuples)


def fiddle():
	print("Loading data...")
	tr_im_image_vector_tuples = wrapper.db_all_filename_img_vec_pairs()
	print("Loaded data.")
	print("First element:,", tr_im_image_vector_tuples[0][0], tr_im_image_vector_tuples[0][1])

if __name__ == "__main__":
	if "fiddle" in sys.argv:
		fiddle()
	elif "norm" in sys.argv:
		normalize_abs_image_vectors()