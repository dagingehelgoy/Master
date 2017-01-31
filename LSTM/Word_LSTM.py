import itertools

import numpy
import sys
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils.visualize_util import plot


def open_corpus():
	sentence_file = open("Flickr8k.token.txt")
	lines = sentence_file.readlines()
	sentence_file.close()
	return lines


def retrieve_sentences(lines):
	# Put sentences in list
	sentences = []
	for line in lines:
		sentence = []
		for x in ((((line.split(".jpg#")[1])[1:]).strip()).split()):
			sentence.append(x.lower())

		sentences.append(sentence)
	# sentences = sentences[:10]
	return sentences


def build_word_dicts(sentences):
	# Give all words an unique id (int), make dictionary with both 'word' : id and id : 'word'
	all_tokens = itertools.chain.from_iterable(sentences)
	word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
	id_to_word = {token: idx for idx, token in word_to_id.items()}
	return word_to_id, id_to_word


def convert_to_index_list(sentences, word_to_id):
	# Add sequences of ids corresponding to the sentences of words
	index_sentences = []
	for sentence in sentences:
		index_sentence = []
		for word in sentence:
			if word in word_to_id:
				index_sentence.append(word_to_id[word])
		index_sentences.append(index_sentence)
	return index_sentences


def create_training_data(index_sentences):
	# Create training data for each word in a sentence as X and all prior words as Y for the whole sentence
	dataX = []
	dataY = []
	for sentence in index_sentences:
		for i in range(1, len(sentence)):
			if sentence[i] != 0:
				x = sentence[0:i]
				dataX.append(x)
				dataY.append(sentence[i])

	dataX = sequence.pad_sequences(dataX)

	dataX = numpy.asarray(dataX)
	dataY = numpy.asarray(dataY)
	return dataX, dataY


def prepare_data(dataX, n_vocab, seq_length):
	n_patterns = len(dataX)
	X = numpy.reshape(dataX, (n_patterns, seq_length - 1, 1))

	# normalize
	X = X / float(n_vocab)

	return X


def batch_generator(dataX, dataY, n_vocab):
	while 1:
		for pos in range(0, len(dataX), 64):
			Xs = []
			ys = []
			for i in range(pos, pos+64):
				X = dataX[pos]
				Xs.append(X)
				y_id = dataY[pos]

				y = numpy.zeros(n_vocab)
				y[y_id] = 1
				ys.append(y)
			yield (numpy.asarray(Xs), numpy.asarray(ys))


def main():
	lines = open_corpus()
	sentences = retrieve_sentences(lines)
	word_to_id, id_to_word = build_word_dicts(sentences)
	index_sentences = convert_to_index_list(sentences, word_to_id)

	# Get longest sentence and additional stats
	n_words = 0
	seq_length = 0
	for sentence in index_sentences:
		n_words += len(sentence)
		if seq_length < len(sentence):
			seq_length = len(sentence)

	print(n_words)

	# Pad sentences with zero to set length of all sentences to the length of the longest
	# pad_index_sentences = sequence.pad_sequences(index_sentences)

	dataX, dataY = create_training_data(index_sentences)
	X = prepare_data(dataX, len(word_to_id), seq_length)

	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(len(word_to_id), activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	train = False
	if train:
		# define the checkpoint
		filepath = "weights-improvement-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		# fit the model
		plot(model, show_layer_names=True, show_shapes=True)
		data_gen = batch_generator(X, dataY, len(word_to_id))
		model.fit_generator(data_gen, len(X), 50, callbacks=callbacks_list)

	else:
		filename = "LSTM/weights-improvement-4.2947.hdf5"
		model.load_weights(filename)

		start_sentence = raw_input("Start sentence... ")
		start_sentence = [x.lower() for x in start_sentence.split(" ")]

		start_sentence = start_sentence[:20]

		x = [word_to_id[word] for word in start_sentence]
		while len(x) < seq_length - 1:
			x = [0] + x

		for i in range(100):

			x = prepare_data([x], len(word_to_id), seq_length)

			preds = model.predict(x, verbose=0)[0]

			argmax = numpy.argmax(preds)
			argmax_normalized = float(argmax) / len(word_to_id)
			x = [numpy.append(x[0][1:], argmax_normalized)]

			sys.stdout.write(id_to_word[argmax] + " ")
			sys.stdout.flush()

main()
