
import sqlite_wrapper as db_wrapper


def fetch_all_word_vectors():
	return db_wrapper.db_fetch_all_word_vectors()


def fetch_word_vector(word, default_return=None):
	return db_wrapper.db_fetch_word_vector(word, default=None)


def save_word_vector(word_text, word_vector):
	return db_wrapper.db_insert_word_vector(word_text, word_vector)

def save_word_vector_tuple(words_tuple):
	return db_wrapper.db_insert_word_vector_list(words_tuple)
