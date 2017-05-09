import itertools

import keras
import numpy
import sys
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Embedding, TimeDistributed
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from keras.utils.visualize_util import plot
from helpers.io_helper import load_pickle_file, check_pickle_file, save_pickle_file
from helpers.list_helpers import print_progress


def open_corpus():
	sentence_file = open("data/datasets/Flickr30k.txt")
	lines = sentence_file.readlines()
	sentence_file.close()
	return lines


def create_training_data(index_sentences, max_seq_length, nb_vocabulary):
	# Create training data for each word in a sentence as X and all prior words as Y for the whole sentence
	dataX = []
	dataY = []
	numpy.random.shuffle(index_sentences)
	for sentence in index_sentences:
		for i in range(1, len(sentence)):
			if sentence[i] != 0:
				x = sentence[0:i]
				dataX.append(x)
				dataY.append(sentence[i])

	dataX = sequence.pad_sequences(dataX, maxlen=max_seq_length)

	dataX = numpy.asarray(dataX)
	dataY = numpy.asarray(dataY)
	return dataX, dataY


def batch_generator(dataX, dataY, n_vocab):
	batch_size = 128
	while 1:
		for pos in range(0, len(dataX), batch_size):
			Xs = []
			ys = []
			for i in range(pos, pos + batch_size):
				X = dataX[i]
				Xs.append(X)
				y_id = dataY[i]
				y = numpy.zeros(n_vocab)
				y[y_id] = 1
				ys.append(y)
			yield (numpy.asarray(Xs), numpy.asarray(ys))


def get_word_embeddings():
	embeddings_index = {}
	f = open('LSTM/glove.6B.300d.txt')
	count = 0
	for line in f:
		values = line.split()
		word = values[0]
		coefs = numpy.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
		count += 1
		if count % 100 == 0:
			print_progress(count, 400000, prefix="Getting glove word embeddings")
	f.close()

	return embeddings_index


def get_model(nb_words, embedding_layer):
	# define the LSTM model
	model = Sequential()
	model.add(embedding_layer)
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(512, return_sequences=False))
	model.add(Dropout(0.5))
	model.add(Dense(nb_words, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model


def word_lstm(train=True):
	NB_WORDS = 20000
	EMBEDDING_DIMENSION = 300
	EPOCHS = 50

	lines = open_corpus()

	######### Tokenizer #########
	lines = [(line.split("\t")[1]).strip() for line in lines]
	print
	lines = ['<SOS> ' + line + ' <EOS>' for line in lines]
	tokenizer = Tokenizer(nb_words=NB_WORDS, filters="""!"#$%&'()*+-/:;=?@[\]^_`{|}~""")
	tokenizer.fit_on_texts(lines)
	sequences = tokenizer.texts_to_sequences(lines)

	word_to_id = tokenizer.word_index
	id_to_word = {token: idx for idx, token in word_to_id.items()}

	sequence_lengths = [len(x) for x in sequences]
	MAX_SEQUENCE_LENGTH = int(numpy.average(sequence_lengths) + numpy.std(sequence_lengths))

	print("Sequence length: %s" % MAX_SEQUENCE_LENGTH)
	print("Nb words: %s" % NB_WORDS)

	######### Embeddings #########
	# embedding_matrix = get_embedding_matrix(word_to_id, EMBEDDING_DIMENSION)
	embedding_layer = Embedding(len(word_to_id) + 1, EMBEDDING_DIMENSION, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=False)

	model = get_model(NB_WORDS, embedding_layer)

	if train:
		# define the checkpoint
		filepath = "trainable_embeddings--weights-{loss:.4f}.hdf5"
		tensorboard = keras.callbacks.TensorBoard(log_dir='text_generators/logs/stacked_word_lstm', histogram_freq=0, write_graph=True, write_images=False)
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint, tensorboard]
		# fit the model
		plot(model, show_layer_names=True, show_shapes=True)

		dataX, dataY = create_training_data(sequences, MAX_SEQUENCE_LENGTH, NB_WORDS)
		data_gen = batch_generator(dataX, dataY, NB_WORDS)
		model.fit_generator(data_gen, len(dataX), EPOCHS, callbacks=callbacks_list)

	else:
		filename = "tokenizer-words-embedding-weights-0.9344.hdf5"
		model.load_weights(filename)

		while True:
			start_sentence = raw_input("\nStart sentence... ")
			start_sentence = [x.lower() for x in start_sentence.split(" ")]

			start_sentence = start_sentence[:20]

			start_sentence = [word_to_id[word] for word in start_sentence]

			for id in start_sentence:
				sys.stdout.write(id_to_word[id] + " ")

			while len(start_sentence) < MAX_SEQUENCE_LENGTH:
				start_sentence = [0] + start_sentence

			start_sentence = [start_sentence]
			for i in range(20):
				# prepared_sentence = prepare_data(start_sentence, tokenizer.nb_words, MAX_SEQUENCE_LENGTH)

				preds = model.predict(numpy.asarray(start_sentence), verbose=0)[0]

				argmax = numpy.argmax(preds)
				# argmax_normalized = float(argmax) / len(word_to_id)
				start_sentence = [numpy.append(start_sentence[0][1:], argmax)]
				sys.stdout.write(id_to_word[argmax] + " ")
				sys.stdout.flush()
				if argmax == 356:
					break

