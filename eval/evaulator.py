import numpy as np
from pyemd import emd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances

from GAN.config import config
from GAN.helpers.datagen import generate_string_sentences
from GAN.helpers.enums import Conf
from bleu import fetch_bleu_score
from eval import tfidf
from helpers.io_helper import load_pickle_file, save_pickle_file
from helpers.list_helpers import insert_and_remove_last, print_progress
from word2vec.word2vec_helpers import get_dict_filename

"""

Word embedding distance

"""


def find_n_most_similar_vectors(pred_vector, vector_list, sentence_list, n=5):
	first_vector = vector_list[0]
	first_sentence = sentence_list[0]
	first_mse = compare_vectors(pred_vector, first_vector)

	best_mse_list = [0 for _ in range(n)]
	best_sentence_list = ["" for _ in range(n)]
	best_vector_list = [[] for _ in range(n)]

	best_mse_list = insert_and_remove_last(0, best_mse_list, first_mse)
	best_vector_list = insert_and_remove_last(0, best_vector_list, first_vector)
	best_sentence_list = insert_and_remove_last(0, best_sentence_list, first_sentence)
	for i in range(len(vector_list)):
		temp_vector = vector_list[i]
		temp_mse = compare_vectors(pred_vector, temp_vector)
		for index in range(len(best_vector_list)):
			if temp_mse < best_mse_list[index]:
				best_mse_list = insert_and_remove_last(index, best_mse_list, temp_mse)
				best_vector_list = insert_and_remove_last(index, best_vector_list, temp_vector)
				best_sentence_list = insert_and_remove_last(index, best_sentence_list, sentence_list[i])
				break
	return best_sentence_list


def convert_to_word_embeddings(sentences, word_embedding_dict):
	embedded_sentences = []
	for sentence in sentences:
		embedded_sentence = []
		words = sentence.split(" ")
		for word in words:
			if word in word_embedding_dict:
				embedded_sentence.append(word_embedding_dict[word])
			else:
				embedded_sentence.append(word_embedding_dict['UNK'])
				# embedded_sentence.append(word_embedding_dict['markus'])

		embedded_sentences.append(embedded_sentence)
	return embedded_sentences


def compare_vectors(v1, v2):
	return cosine(v1, v2)


def convert_vectors(vectors):
	sum_vector = np.zeros(vectors[0].shape)
	for word_emb in vectors:
		sum_vector += word_emb
	return sum_vector


def convert_to_emb_list(dataset_string_list_sentences, word_embedding_dict):
	dataset_emb_list_sentences = []
	for sentence in dataset_string_list_sentences:
		s = []
		for word in sentence:
			if word in word_embedding_dict:
				s.append(word_embedding_dict[word])
			else:
				s.append(word_embedding_dict['UNK'])
				# s.append(word_embedding_dict['markus'])
		dataset_emb_list_sentences.append(s)
	return dataset_emb_list_sentences


def cosine_distance_retrieval(pred_strings, dataset_string_list_sentences, word_embedding_dict):
	dataset_emb_list_sentences = convert_to_emb_list(dataset_string_list_sentences, word_embedding_dict)
	dataset_single_vector_sentences = [convert_vectors(sentence) for sentence in dataset_emb_list_sentences]
	pred_emb_list_sentences = convert_to_word_embeddings(pred_strings, word_embedding_dict)
	pred_single_vector_sentences = [convert_vectors(sentence) for sentence in pred_emb_list_sentences]

	best_sentence_lists = []
	tot_count = len(pred_single_vector_sentences)
	for i in range(tot_count):
		pred_single_vector_sentence = pred_single_vector_sentences[i]
		best_sentence_list = find_n_most_similar_vectors(pred_single_vector_sentence, dataset_single_vector_sentences,
														 dataset_string_list_sentences)
		best_sentence_lists.append([" ".join(x) for x in best_sentence_list])
		print_progress(i + 1, tot_count, "Calculating cosine distances")
	return best_sentence_lists


"""

TF-IDF

"""


def tfidf_retrieval(pred_strings, dataset_string_list_sentences):
	table = tfidf.tfidf()
	for dataset_entry in dataset_string_list_sentences:
		table_name = " ".join(dataset_entry)
		table.addDocument(table_name, [str(x) for x in dataset_entry])
	best_sentence_lists = []
	for pred_string in pred_strings:
		similarities = table.similarities(pred_string.split(" "))
		similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
		best_sentence_lists.append([x[0] for x in similarities[:5]])
	return best_sentence_lists


"""

Word bower distance

"""


def get_wmd_distance(d1, d2, word_embedding_dict, min_vocab=7, verbose=False):
	model_vocab = word_embedding_dict.keys()
	vocabulary = [w for w in set(d1.lower().split() + d2.lower().split()) if
				  w in model_vocab and w not in stop_words.ENGLISH_STOP_WORDS]
	if len(vocabulary) < min_vocab:
		return 1
	vect = CountVectorizer(vocabulary=vocabulary).fit([d1, d2])
	feature_names = vect.get_feature_names()
	W_ = np.array([word_embedding_dict[w] for w in feature_names if w in word_embedding_dict])
	D_ = euclidean_distances(W_)
	D_ = D_.astype(np.double)
	D_ /= D_.max()  # just for comparison purposes

	d1 = d1.split(" ")
	d2 = d2.split(" ")
	v_1 = [0.0 for _ in range(len(feature_names))]
	v_2 = [0.0 for _ in range(len(feature_names))]
	for i in range(len(feature_names)):
		voc = feature_names[i]
		v_1[i] += d1.count(voc)
		v_2[i] += d2.count(voc)


	# v_1, v_2 = vect.transform([d1, d2])
	# v_1 = v_1.toarray().ravel()
	# v_2 = v_2.toarray().ravel()
	# pyemd needs double precision input
	# v_1 = v_1.astype(np.double)
	# v_2 = v_2.astype(np.double)
	v_1 = np.asarray(v_1)
	v_2 = np.asarray(v_2)
	v_1 /= v_1.sum()
	v_2 /= v_2.sum()
	if verbose:
		print vocabulary
		print v_1, v_2
	return emd(v_1, v_2, D_)


def wmd_retrieval(pred_strindgs, dataset_string_list_sentences):
	filename = get_dict_filename(config[Conf.EMBEDDING_SIZE], config[Conf.WORD2VEC_NUM_STEPS],
								 config[Conf.VOCAB_SIZE], config[Conf.W2V_SET])
	word_embedding_dict = load_pickle_file(filename)

	best_sentence_lists = []

	for pred_string in pred_strings:

		score_tuples = []
		for dataset_string_list_sentence in dataset_string_list_sentences:
			dataset_string = " ".join(dataset_string_list_sentence)
			score = get_wmd_distance(pred_string, dataset_string, word_embedding_dict)
			score_tuples.append((dataset_string, score))
		score_tuples = sorted(score_tuples, key=lambda x: x[1], reverse=False)
		result = [x[0] for x in score_tuples[:5]]

		best_sentence_lists.append(result)

	return best_sentence_lists


from multiprocessing import Pool, Value
import multiprocessing

counter = None

def init(args):
	''' store the counter for later use '''
	global counter
	counter = args



def background_wmd_retrieval(pred_strings, dataset_string_list_sentences):
	filename = get_dict_filename(config[Conf.EMBEDDING_SIZE], config[Conf.WORD2VEC_NUM_STEPS], config[Conf.VOCAB_SIZE], config[Conf.W2V_SET])
	word_embedding_dict = load_pickle_file(filename)

	counter = Value('i', 0)

	cpu_count = multiprocessing.cpu_count()
	print "CPUs:", cpu_count
	if cpu_count > 8:
		cpu_count = 10
	pool = Pool(cpu_count, initializer=init, initargs=(counter, ))
	tuple_array = [(pred_string, dataset_string_list_sentences, word_embedding_dict) for pred_string in pred_strings]
	best_sentence_lists = pool.map(background_wmd, tuple_array, chunksize=1)
	pool.close()
	pool.join()

	return best_sentence_lists


def background_wmd(tuple):
	global counter

	pred_string, dataset_string_list_sentences, word_embedding_dict = tuple
	score_tuples = []
	for dataset_string_list_sentence in dataset_string_list_sentences:
		dataset_string = " ".join(dataset_string_list_sentence)
		score = get_wmd_distance(pred_string, dataset_string, word_embedding_dict)
		score_tuples.append((dataset_string, score))
	score_tuples = sorted(score_tuples, key=lambda x: x[1], reverse=False)
	result = [x[0] for x in score_tuples[:5]]
	with counter.get_lock():
		counter.value += 1

	print_progress(counter.value, 1000, "Running WMD")
	# print counter.value
	return result


from collections import Counter
def calculate_bleu_score(sentences, dataset_string_list_sentences=None, word_embedding_dict=None):
	# print "Evaluating %s generated sentences." % len(sentences)
	if dataset_string_list_sentences is None or word_embedding_dict is None:
		if not config[Conf.LIMITED_DATASET].endswith("_uniq.txt"):
			config[Conf.LIMITED_DATASET] = config[Conf.LIMITED_DATASET].split(".txt")[0] + "_uniq.txt"
		dataset_string_list_sentences, word_embedding_dict = generate_string_sentences(config)

	count_dict = Counter(sentences)
	uniq_sentences = count_dict.keys()
	print "Finding reference sentneces using WMD"
	best_sentence_lists_wmd = background_wmd_retrieval(uniq_sentences, dataset_string_list_sentences)

	print "Finding reference sentneces using cosine distance"
	best_sentence_lists_cosine = cosine_distance_retrieval(uniq_sentences, dataset_string_list_sentences,
	                                                       word_embedding_dict)

	print "Finding reference sentneces using TF-IDF"
	best_sentence_lists_tfidf = tfidf_retrieval(uniq_sentences, dataset_string_list_sentences)

	bleu_score_tot_cosine = 0
	bleu_score_tot_tfidf = 0
	bleu_score_tot_wmd = 0
	sentence_score_dict = {}
	for i in range(len(uniq_sentences)):
		sentence = uniq_sentences[i]

		bleu_cosine = fetch_bleu_score(best_sentence_lists_cosine[i], sentence)
		bleu_tfidf = fetch_bleu_score(best_sentence_lists_tfidf[i], sentence)
		bleu_wmd = fetch_bleu_score(best_sentence_lists_wmd[i], sentence)

		bleu_score_tot_cosine += (bleu_cosine * count_dict[sentence])
		bleu_score_tot_tfidf += (bleu_tfidf * count_dict[sentence])
		bleu_score_tot_wmd += (bleu_wmd * count_dict[sentence])

		beta_score = (bleu_cosine + bleu_tfidf + bleu_wmd) / 3

		sentence_score_dict[sentence] = (beta_score, count_dict[sentence])

	avg_bleu_tfidf = bleu_score_tot_tfidf / float(len(sentences))
	avg_bleu_wmd = bleu_score_tot_wmd / float(len(sentences))
	avg_bleu_cosine = bleu_score_tot_cosine / float(len(sentences))
	save_pickle_file(sentence_score_dict, "emb_score_dict.p")
	avg_bleu_score = (avg_bleu_cosine + avg_bleu_tfidf + avg_bleu_wmd) / 3

	# print "BLEU score cosine:\t", avg_bleu_cosine
	# print "BLEU score tfidf:\t", avg_bleu_tfidf
	# print "BLEU score wmd:\t\t", avg_bleu_wmd
	# print "Avarage BLEU score:", avg_bleu_score
	print "AVG BLEU: %5.4f\t %5.4f,%5.4f,%5.4f (cosine,tfidf,wmd)" % (
	avg_bleu_score, avg_bleu_cosine, avg_bleu_tfidf, avg_bleu_wmd)
	return avg_bleu_score, avg_bleu_cosine, avg_bleu_tfidf, avg_bleu_wmd


def eval_main():
	eval_dataset_string_list_sentences, eval_word_embedding_dict = generate_string_sentences(config)
	sentences = ["<sos> the flower har large green petals and black stamen <eos> <pad>",
					"<sos> this flower has yellow petals and middle red stamen <eos> <pad>",
					"<sos> this flower has many yellow petals with yellow stamen <eos> <pad>",
					"<sos> stamens are yellow in color with larger anthers <eos> <pad> <pad>"]

	sentences = [
		# "<sos> the flower har large green petals and black stamen <eos> <pad>",
		# "<sos> this flower has yellow petals and middle red stamen <eos> <pad>",
		# "<sos> this flower has many yellow petals with yellow stamen <eos> <pad>",
		# "<sos> stamens are yellow in color with larger anthers <eos> <pad> <pad>"
		# "this flower has petals that are yellow with white edges"
		# "<sos> this flower has petals that are blue with blue stamen <eos>"
		# "<sos> this flower has blue petals with with with green stamen <eos>"
		# "<sos> this flower has petals that are with with white edges <eos>"
		"< plot raincoat petals attractions wielded fuchsia gently nut ol = ="
		# "< spoilers precocial classify remiss 8-space interchanged interchanged suffice tenchu = ntagtop"
		# "<sos> this flower has petals that are yellow with white edges <eos>"
		# "<sos> 2 villagers carry a baskets of goods while another follows <eos>"
		# "<sos> a young girl on the beach running toward the water <eos>"
		# "<sos> Two boys in blue shirts wearing backpacks <eos> <pad> <pad> <pad>"
		# "<sos> Two soccer players swim on the soccer field <eos> <pad> <pad>"
		# "<sos> An old jeep partially submerged in water <eos> <pad> <pad> <pad>"
		# "<sos> a man is climbing up a wall <eos> <pad> <pad> <pad>"
		# "<sos> five sided white flower flower <eos> <pad> <pad> <pad> <pad> <pad>"
	]

	best_sentence_lists_cosine = cosine_distance_retrieval(sentences, eval_dataset_string_list_sentences, eval_word_embedding_dict)

	best_sentence_lists_tfidf = tfidf_retrieval(sentences, eval_dataset_string_list_sentences)

	best_sentence_lists_wmd = background_wmd_retrieval(sentences, eval_dataset_string_list_sentences)

	print
	print "Sentence:"
	print sentences[0]
	print
	print "COS"
	for cos in best_sentence_lists_cosine[0][:5]:
		print cos
	print
	print "TFIDF"
	for tf in best_sentence_lists_tfidf[0][:5]:
		print tf
	print
	print "WMD"
	for wmd in best_sentence_lists_wmd[0][:5]:
		print wmd

	calculate_bleu_score(sentences, eval_dataset_string_list_sentences, eval_word_embedding_dict)


def eval_seqgan():
	eval_dataset_string_list_sentences, eval_word_embedding_dict = generate_string_sentences(config)

	seqgan_file = open("eval/files/seqgan_flower.txt")
	seqgan_lines = seqgan_file.readlines()[:1000]
	seqgan_file.close()
	seqgan_lines = [x.strip() for x in seqgan_lines]
	distinct_sentences = len(set(seqgan_lines))
	sentence_count = len(seqgan_lines)
	calculate_bleu_score(seqgan_lines, eval_dataset_string_list_sentences, eval_word_embedding_dict)
	print "Number of distict sentences: %s/%s" % (distinct_sentences, sentence_count)


if __name__ == '__main__':
	eval_seqgan()
	# eval_main()
