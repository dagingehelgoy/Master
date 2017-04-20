# encoding=utf8

from data.database.helpers.caption_database_helper import *
from data.database.helpers.image_database_helper import *
from data.database.helpers.pca_database_helper import fetch_all_pca_vector_pairs
from helpers.io_helper import *


def fetch_embeddings(size=-1):
	if embedding_exists(size):
		return load_embeddings(size)
	else:
		print("Generating compatible dataset...")
		all_image_names, image_name_caption_dict = create_dictionaries(size)
		image_names, image_data, image_captions = get_examples(all_image_names, image_name_caption_dict)

		dataset = [image_names, np.asarray(image_data), image_captions]
		print("Finished generating %s training example" % len(image_captions))
		save_embeddings(dataset, size)

		return dataset


def create_dictionaries(size):
	if size > 0:
		all_image_names = fetch_all_image_names()[:size]
	else:
		all_image_names = fetch_all_image_names()
	num_images = len(all_image_names)
	validate_database(num_images)
	image_name_caption_dict = dict()
	name_cap_tuples = fetch_all_caption_text_tuples()
	for (name, caption) in name_cap_tuples:
		if name in image_name_caption_dict:
			image_name_caption_dict[name].append(caption)
		else:
			image_name_caption_dict[name] = [caption]
	return all_image_names, image_name_caption_dict


def get_similarity_dictionary():
	pickle_file = open(settings.ROOT_DIR + "helpers/similarity-dict.p", 'rb')
	dataset = pickle.load(pickle_file)
	pickle_file.close()
	return dataset


def get_examples(all_image_names, image_name_caption_vector_dict):
	sorted_caption_vector_data = []
	sorted_image_data = []
	sorted_image_names = []
	image_name_pca_vector_dict = {key: value for (key, value) in fetch_all_pca_vector_pairs()}
	all_image_names_total = len(all_image_names)
	for i in range(all_image_names_total):
		image_name = all_image_names[i]
		pca_vector = image_name_pca_vector_dict[image_name]
		caption_vectors = image_name_caption_vector_dict[image_name]

		for caption_vector in caption_vectors:
			sorted_image_data.append(pca_vector)
			sorted_image_names.append(image_name)
			sorted_caption_vector_data.append(caption_vector)
		print_progress(i + 1, all_image_names_total, prefix='Generating data:', suffix='Complete', barLength=50)

	return sorted_image_names, sorted_image_data, sorted_caption_vector_data


def get_class_examples(image_name_class_vector_dict):
	sorted_class_vector_data = []
	sorted_image_data = []
	image_vector_dict = {key: value for (key, value) in fetch_all_image_vector_pairs()}
	all_image_names_total = len(image_name_class_vector_dict.keys())
	counter = 1
	for key in image_name_class_vector_dict:
		image_vector = image_vector_dict[key]
		class_vectors = image_name_class_vector_dict[key]
		for class_vector in class_vectors:
			sorted_image_data.append(image_vector)
			sorted_class_vector_data.append(class_vector)
		counter += 1
		print_progress(counter, all_image_names_total, prefix='Generating data:', suffix='Complete', barLength=50)

	return sorted_class_vector_data, sorted_image_data


def save_embeddings(dataset_to_store, size):
	filepath = find_stored_embeddings_filepath(size)
	save_pickle_file(dataset_to_store, filepath)


def save_class_embeddings(dataset_to_store, size):
	filepath = find_stored_class_embeddings_filepath(size)
	save_pickle_file(dataset_to_store, filepath)


def load_embeddings(size):
	print("Loading compatible dataset from local storage: %s" % get_stored_embeddings_filename(size))
	filepath = find_stored_embeddings_filepath(size)
	return load_pickle_file(filepath)


def load_class_embeddings(size):
	print("Loaded compatible dataset from local storage: %s" % get_stored_class_embeddings_filename(size))
	filepath = find_stored_class_embeddings_filepath(size)
	return load_pickle_file(filepath)


def find_stored_embeddings_filepath(size):
	return settings.STORED_EMBEDDINGS_DIR + get_stored_embeddings_filename(size)


def find_stored_class_embeddings_filepath(size):
	return settings.STORED_EMBEDDINGS_DIR + get_stored_class_embeddings_filename(size)


def embedding_exists(size):
	filepath = find_stored_embeddings_filepath(size)
	return check_pickle_file(filepath)


def class_embedding_exists(size):
	filepath = find_stored_class_embeddings_filepath(size)
	return check_pickle_file(filepath)


def validate_database(num_images):
	if num_images == 0:
		raise IOError('No images in databases')
	elif fetch_caption_count() == 0:
		raise IOError('No captions in databases')


def get_stored_embeddings_filename(size):
	if size == -1:
		size = "all"
	return "%s-%s.picklefile" % (settings.STORED_EMBEDDINGS_NAME, size)


def get_stored_class_embeddings_filename(size):
	if size == -1:
		size = "all"
	return "class-%s-%s.picklefile" % (settings.STORED_EMBEDDINGS_NAME, size)


if __name__ == "__main__":
	from image_database_helper import fetch_filename_from_image_vector
	from caption_database_helper import fetch_filenames_from_cation_vector, fetch_filename_caption_tuple, \
		fetch_all_filename_caption_vector_tuples

	dataset = fetch_embeddings(5)
	caption_vectors = dataset[0]
	img_vecs = dataset[1]
	sims = dataset[2]
	print("%s \t %s \t %s \t %s" % (
		"filename from cap", "filename from img", "similarity",
		"text caption"))
	for i in range(len(caption_vectors)):
		cap_vec = caption_vectors[i]
		img_vec = img_vecs[i]
		sim = sims[i]

		print("%s \t %s \t %s \t %s" % (
			fetch_filenames_from_cation_vector(cap_vec), fetch_filename_from_image_vector(img_vec), sim,
			fetch_filename_caption_tuple(cap_vec)[1]))
