import os
import sys

file_par_dir = os.path.join(__file__, os.pardir)
file_par_par_dir = os.path.join(file_par_dir, os.pardir)
file_par_par_par_dir = os.path.join(file_par_par_dir, os.pardir)
ROOT_DIR = os.path.dirname((os.path.abspath(file_par_par_par_dir))) + "/"
sys.path.append(ROOT_DIR)

import sqlite_wrapper as wrapper


def store_pca_vector_to_db(image_name, vector):
	wrapper.db_insert_pca_vector(image_name, vector)


def fetch_pca_vector(image_name):
	return wrapper.db_get_pca_vector(image_name)[0]


def fetch_all_pca_vector_pairs():
	return wrapper.db_all_filename_pca_vec_pairs()


def fetch_filename_from_pca_vector(pca_vector):
	return wrapper.db_get_filename_from_pca_vector(pca_vector)


def update_pca_vectors(filename_pca_vector_tuples):
	return wrapper.db_insert_pca_vector_list(filename_pca_vector_tuples)
