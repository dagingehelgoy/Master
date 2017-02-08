from list_helpers import insert_and_remove_last

import sqlite_wrapper as db_wrapper
from data.database.helpers.caption_database_helper import fetch_all_caption_text_tuples, compare_vectors
from data.database.helpers.word_database_helper import fetch_all_word_vectors
from helpers.io_helper import save_pickle_file, load_pickle_file
from helpers.list_helpers import print_progress


def save_class_vector(image_name, class_text, class_vector):
	db_wrapper.db_insert_class_vector(image_name, class_text, class_vector)


def save_class_vector_list(tuple_list):
	print("Storing class in data...")
	db_wrapper.db_insert_class_vector_list(tuple_list)


def fetch_class_vectors_for_image_name(image_name):
	vectors = db_wrapper.db_get_class_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def fetch_class_texts_for_image_name(image_name):
	vectors = db_wrapper.db_get_class_texts(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def fetch_all_class_vectors():
	return db_wrapper.db_fetch_all_class_vectors()


def fetch_filename_class_tuple(class_vector):
	return db_wrapper.db_get_filename_class_tuple_from_class_vector(class_vector)


def fetch_class_count():
	return db_wrapper.db_get_class_table_size()


def fetch_all_filename_class_vector_tuples():
	return db_wrapper.db_all_filename_class_vector_tuple()


def fetch_all_class_rows():
	return db_wrapper.db_all_class_rows()


def fetch_filenames_from_cation_vector(class_vector):
	return db_wrapper.db_get_filenames_from_class_vector(class_vector)


if __name__ == "__main__":
	caps = fetch_class_vectors_for_image_name("1000092795.jpg")
	for cap in caps:
		print len(cap)


def create_common_words_pickle():
	common_words_file = open('common_words.txt', 'r')
	common_words = common_words_file.readlines()
	common_words_file.close()
	words = []
	for common_word in common_words:
		for x in common_word.split('\r'):
			lower = x.strip().lower()
			words.append(lower)
	save_pickle_file(words, "common_words.p")


def get_classes(caption, common_words):
	words = caption.split(" ")
	classes = []
	for word in words:
		word = word.lower()
		if word not in common_words and len(word) > 2:
			classes.append(word)
	return classes


def find_n_most_similar_class(class_embedding, n=1):
	class_vector_pairs = fetch_all_word_vectors()

	first_class_text = class_vector_pairs[0][0]
	first_class_vector = class_vector_pairs[0][1]
	first_class_mse = compare_vectors(class_embedding, first_class_vector)

	best_class_vector_mse_list = [0 for i in range(n)]
	best_class_text_list = ["" for i in range(n)]
	best_class_vector_list = [[] for i in range(n)]

	best_class_vector_mse_list = insert_and_remove_last(0, best_class_vector_mse_list, first_class_mse)
	best_class_text_list = insert_and_remove_last(0, best_class_text_list, first_class_text)
	best_class_vector_list = insert_and_remove_last(0, best_class_vector_list, first_class_vector)
	total_classs = len(class_vector_pairs)
	counter = 1

	print_progress(counter, total_classs, prefix="Searching for class")
	for temp_class_text, temp_class_vector in class_vector_pairs:
		temp_image_mse = compare_vectors(class_embedding, temp_class_vector)
		for index in range(len(best_class_vector_list)):
			if temp_image_mse < best_class_vector_mse_list[index]:
				best_class_vector_mse_list = insert_and_remove_last(index, best_class_vector_mse_list, temp_image_mse)
				best_class_text_list = insert_and_remove_last(index, best_class_text_list, temp_class_text)
				best_class_vector_list = insert_and_remove_last(index, best_class_vector_list, temp_class_vector)
				break
		counter += 1
		if counter % 100 == 0 or counter > total_classs - 1:
			print_progress(counter, total_classs, prefix="Searching for class")
	print_progress(total_classs, total_classs, prefix="Searching for class")
	return best_class_text_list


def gen_class_embs():
	# create_common_words_pickle()
	print "Generating classes"
	common_words = load_pickle_file("common_words.p")
	print "Loading captions..."
	filename_caption_text_tuples = fetch_all_caption_text_tuples()[:5000]
	print "Loading word embeddings..."
	word_embedding_dict = dict(fetch_all_word_vectors())
	filname_text_vector_tuples = []
	tot = len(filename_caption_text_tuples)
	counter = 1
	print_progress(counter, tot, prefix="Converting classes to embs")
	for filename, caption in filename_caption_text_tuples:
		classes = get_classes(caption, common_words)
		filname_text_vector_tuples.extend([(filename, c, word_embedding_dict[c]) for c in classes if c in word_embedding_dict.keys()])
		counter += 1
		print_progress(counter, tot, prefix="Converting classes to embs")

	save_class_vector_list(filname_text_vector_tuples)

