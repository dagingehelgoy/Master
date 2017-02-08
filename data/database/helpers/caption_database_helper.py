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
	vectors = db_wrapper.db_get_caption_texts(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


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


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def find_n_most_similar_captions(caption_embedding, n=1):
	caption_vector_pairs = fetch_all_caption_rows()

	first_caption_vector = caption_vector_pairs[0][1]
	first_caption_text = caption_vector_pairs[0][2]
	first_caption_mse = compare_vectors(caption_embedding, first_caption_vector)

	best_caption_vector_mse_list = [0 for i in range(n)]
	best_caption_text_list = ["" for i in range(n)]
	best_caption_vector_list = [[] for i in range(n)]

	best_caption_vector_mse_list = insert_and_remove_last(0, best_caption_vector_mse_list, first_caption_mse)
	best_caption_text_list = insert_and_remove_last(0, best_caption_text_list, first_caption_text)
	best_caption_vector_list = insert_and_remove_last(0, best_caption_vector_list, first_caption_vector)
	total_captions = len(caption_vector_pairs)
	counter = 1

	print_progress(counter, total_captions, prefix="Searching for caption")
	for temp_caption_filename, temp_caption_vector, temp_caption_text in caption_vector_pairs:
		temp_image_mse = compare_vectors(caption_embedding, temp_caption_vector)
		for index in range(len(best_caption_vector_list)):
			if temp_image_mse < best_caption_vector_mse_list[index]:
				best_caption_vector_mse_list = insert_and_remove_last(index, best_caption_vector_mse_list, temp_image_mse)
				best_caption_text_list = insert_and_remove_last(index, best_caption_text_list, temp_caption_text)
				best_caption_vector_list = insert_and_remove_last(index, best_caption_vector_list, temp_caption_vector)
				break
		counter += 1
		if counter % 100 == 0 or counter > total_captions - 1:
			print_progress(counter, total_captions, prefix="Searching for caption")
	print_progress(total_captions, total_captions, prefix="Searching for caption")
	return best_caption_text_list


if __name__ == "__main__":
	caps = fetch_caption_vectors_for_image_name("1000092795.jpg")
	for cap in caps:
		print len(cap)
