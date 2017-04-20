from sklearn.metrics import mean_squared_error

import sqlite_wrapper as db_wrapper
from helpers.list_helpers import insert_and_remove_last, print_progress


def save_caption_vector(image_name, caption_text, caption_vector):
	db_wrapper.db_insert_caption_vector(image_name, caption_text, caption_vector)


def save_caption_vector_list(tuple_list):
	print("Storing captions in data...")
	db_wrapper.db_insert_caption_vector_list(tuple_list)


def fetch_caption_vectors_for_image_name(image_name):
	vectors = db_wrapper.db_get_caption_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def fetch_caption_texts_for_image_name(image_name):
	texts = db_wrapper.db_get_caption_texts(image_name)
	text_list = [i[0] for i in texts]
	return text_list


def fetch_all_caption_vectors():
	return db_wrapper.db_fetch_all_caption_vectors()


def fetch_filename_caption_tuple(caption_vector):
	return db_wrapper.db_get_filename_caption_tuple_from_caption_vector(caption_vector)


def fetch_caption_count():
	return db_wrapper.db_get_caption_table_size()


def fetch_all_filename_caption_vector_tuples():
	return db_wrapper.db_all_filename_caption_vector_tuple()


def fetch_all_caption_rows():
	return db_wrapper.db_all_caption_rows()


def fetch_filenames_from_cation_vector(caption_vector):
	return db_wrapper.db_get_filenames_from_caption_vector(caption_vector)


def fetch_all_caption_text_tuples():
	return db_wrapper.db_all_caption_text_tuples()


def store_caption_text_to_db():
	text_file_path = "/Users/markus/workspace/master/Master/data/datasets/Flickr8k.txt"
	text_file = open(text_file_path)
	lines = text_file.readlines()
	text_file.close()
	captions = []
	for line in lines:
		image_name = line.split("#")[0]
		caption_text = ((line.split("#")[1])[1:]).strip()
		captions.append((image_name, caption_text, None))
	save_caption_vector_list(captions)


if __name__ == "__main__":
	store_caption_text_to_db()
