import collections
import random

import numpy as np

from sklearn.manifold import TSNE

from helpers.io_helper import save_pickle_file, load_pickle_file
from helpers.list_helpers import print_progress
from helpers.text_preprocessing import preprocessing


def read_flickr_data():
	index = 0
	filename = 'data/datasets/Flickr30k.txt'
	"""Extract the first file enclosed in a zip file as a list of words"""
	with open(filename) as f:
		data = []
		readlines = f.readlines()
		length = len(readlines)
		for line in readlines:
			index += 1
			sentence = ((line.split("#")[1])[1:]).strip()
			preprocessed_words = preprocessing(sentence, add_sos_eos=True)
			for x in preprocessed_words:
				data.append(x)
			if index % 10000 == 0:
				print_progress(index, length, prefix='Read data:', suffix='Complete', barLength=50)
	return data


def read_flower_data():
	index = 0
	filename = 'data/datasets/all_flowers.txt'
	"""Extract the first file enclosed in a zip file as a list of words"""
	with open(filename) as f:
		data = []
		readlines = f.readlines()
		length = len(readlines)
		for line in readlines:
			index += 1
			sentence = line.strip()
			preprocessed_words = preprocessing(sentence, add_sos_eos=True)
			for x in preprocessed_words:
				data.append(x)
			if index % 10000 == 0:
				print_progress(index, length, prefix='Read data:', suffix='Complete', barLength=50)
	return data


def build_dataset(vocabulary_size, dataset):
	# Read the datasets into a list of strings.
	if dataset == 'flickr':
		words = read_flickr_data()
	else:
		words = read_flower_data()
	print('Data size', len(words))
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window, data_index):
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=batch_size, dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window  # target label at the center of the buffer
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels, data_index


def plot_with_labels(reverse_dictionary, final_embeddings, filename='tsne', plot_only=500):
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	labels = [reverse_dictionary[i] for i in range(plot_only)]
	import matplotlib.pyplot as plt
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig("word2vec/saved_models/%s.png" % filename)


def plot_with_labels_selected(reverse_dictionary, final_embeddings, selected_word_list, filename='tsne', plot_only=500):
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	labels = [reverse_dictionary[i] for i in range(plot_only)]
	import matplotlib.pyplot as plt
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		if label in selected_word_list:
			x, y = low_dim_embs[i, :]
			plt.scatter(x, y)
			plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig("word2vec/saved_models/%s.png" % filename)


def plot_collections(collection_list, collection_list_names, perplexity, suffix, use_legend=False):
	from matplotlib.pyplot import legend
	print "Plot collections"
	import matplotlib.pyplot as plt
	plt.figure(figsize=(18, 18))  # in inches
	# colors = ['#F95400', '#F9C000', '#004FA2', 'y']
	colors = ['#F9C000', '#004FA2']
	for i in range(len(collection_list)):
		plt.scatter(collection_list[i][:, 0], collection_list[i][:, 1], c=colors[i], label=collection_list_names[i])
	if use_legend:
		legend()
	plot_filename = "word2vec/plots/comparison_"
	for i in range(len(collection_list)):
		plot_filename += collection_list_names[i] + "_"
	plt.savefig("%s%s%s" % (plot_filename, str(perplexity), suffix))
	print "Plot collections done"


def save_model(reverse_dictionary, embeddings, embedding_size, vocab_size, num_steps, dataset):
	word_embeddings_dict = {}
	count = 0
	for word_text, word_vector in zip(reverse_dictionary, embeddings):
		count += 1
		word_embeddings_dict[reverse_dictionary[word_text]] = word_vector
	print count
	dict_filename = get_dict_filename(embedding_size, num_steps, vocab_size, dataset)
	reverse_filename = get_reverse_filename(embedding_size, num_steps, vocab_size, dataset)
	embeddings_filename = get_embeddin_filename(embedding_size, num_steps, vocab_size, dataset)
	save_pickle_file(word_embeddings_dict, dict_filename)
	save_pickle_file(reverse_dictionary, reverse_filename)
	save_pickle_file(embeddings, embeddings_filename)


def load_model(embedding_size, vocab_size, num_steps, dataset):
	reverse_filename = get_reverse_filename(embedding_size, num_steps, vocab_size, dataset)
	embeddings_filename = get_embeddin_filename(embedding_size, num_steps, vocab_size, dataset)
	reverse_dictionary = load_pickle_file(reverse_filename)
	final_embeddings = load_pickle_file(embeddings_filename)
	dict_filename = get_dict_filename(embedding_size, num_steps, vocab_size, dataset)
	dictionary = load_pickle_file(dict_filename)
	return reverse_dictionary, final_embeddings, dictionary


def get_dict_filename(embedding_size, num_steps, vocab_size, dataset):
	return "word2vec/saved_models/word2vec_%sd%svoc%ssteps_dict_%s.pkl" % (embedding_size, vocab_size, num_steps, dataset)


def get_embeddin_filename(embedding_size, num_steps, vocab_size, dataset):
	return "word2vec/saved_models/word2vec_%sd%svoc%ssteps_embs_%s.pkl" % (embedding_size, vocab_size, num_steps, dataset)


def get_reverse_filename(embedding_size, num_steps, vocab_size, dataset):
	return "word2vec/saved_models/word2vec_%sd%svoc%ssteps_reverse_%s.pkl" % (embedding_size, vocab_size, num_steps, dataset)
