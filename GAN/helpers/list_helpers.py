import sys

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=30):
	formatStr = "{0:." + str(decimals) + "f}"
	percents = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def insert_and_remove_last(index, array, element):
	array.insert(index, element)
	del array[-1]
	return array


def find_n_most_similar_images(vector, word_list, vector_list, n=5):
	first_vector = vector_list[0]
	first_word = word_list[0]
	first_mse = compare_vectors(vector, first_vector)

	best_mse_list = [0 for _ in range(n)]
	best_word_list = ["" for _ in range(n)]
	best_vector_list = [[] for _ in range(n)]

	best_mse_list = insert_and_remove_last(0, best_mse_list, first_mse)
	best_word_list = insert_and_remove_last(0, best_word_list, first_word)
	best_vector_list = insert_and_remove_last(0, best_vector_list, first_vector)
	for i in range(len(word_list)):
		temp_word = word_list[i]
		temp_vector = vector_list[i]
		temp_mse = compare_vectors(vector, temp_vector)
		for index in range(len(best_vector_list)):
			if temp_mse < best_mse_list[index]:
				best_mse_list = insert_and_remove_last(index, best_mse_list, temp_mse)
				best_word_list = insert_and_remove_last(index, best_word_list, temp_word)
				best_vector_list = insert_and_remove_last(index, best_vector_list, temp_vector)
				break
	return best_word_list, best_vector_list


def pairwise_cosine_similarity(predicted_word_vectors, embedding_dictionary):
	word_vectors = embedding_dictionary.values()
	word_strings = embedding_dictionary.keys()

	cos_dis_matrix = cosine_similarity(predicted_word_vectors, word_vectors)

	predicted_word_vector_count = len(predicted_word_vectors)

	all_word_vectors_count = len(word_vectors)
	most_similar_words_list = []
	for predicted_image_index in range(predicted_word_vector_count):
		similarities = []
		for glove_vector_index in range(all_word_vectors_count):
			glove_word = word_strings[glove_vector_index]
			cos_sim = cos_dis_matrix[predicted_image_index][glove_vector_index]
			similarities.append((glove_word, cos_sim))
		similarities.sort(key=lambda s: s[1], reverse=True)
		most_similar_words = [x[0] for x in similarities[:5]]
		most_similar_words_list.append(most_similar_words)

	return most_similar_words_list


if __name__ == '__main__':
	# all_word_vectors = np.asarray([[.1, .1], [.2, .2], [.4, .4], [.5, .5]])
	glove_dict_mock = {
		"a": [1, 2],
		"b": [3, 4],
		"c": [5, 6],
		"d": [7, 8]
	}
	predicted_vectors = [
		[1, 2],
		[3, 4]
	]
	most_sim_words_list = pairwise_cosine_similarity(predicted_vectors, glove_dict_mock)
	for x in predicted_vectors:
		print x
	for x in most_sim_words_list:
		print x
