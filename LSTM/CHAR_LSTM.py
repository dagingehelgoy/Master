import argparse
import itertools
import helpers
import pickle
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
import sys


def open_sentences():
	sentence_file = open("../Flickr8k.token.txt")
	lines = sentence_file.readlines()
	sentence_file.close()
	return lines


def extract_sentences(lines):
	# Put sentences in list
	sentences = []
	for line in lines:
		sentence = []
		for x in (((line.split(".jpg#")[1])[1:]).strip()):
			sentence.append(x.lower())
		sentences.append(sentence)
	return sentences


def main(train=True):
	lines = open_sentences()
	sentences = extract_sentences(lines)

	all_tokens_iterator = itertools.chain.from_iterable(sentences)
	all_tokens = sorted(list(set(all_tokens_iterator)))

	char_to_int = dict((c, i) for i, c in enumerate(all_tokens))
	int_to_char = dict((i, c) for i, c in enumerate(all_tokens))
	# summarize the loaded data
	n_chars = sum(len(x) for x in sentences)
	n_vocab = len(all_tokens)
	print "Total Characters: ", n_chars
	print "Total Vocab: ", n_vocab

	dataX = []
	dataY = []
	for sentence in sentences:
		for i in range(1, len(sentence)):
			char_sequence_input = sentence[0:i]
			char_sequence_output = char_to_int[sentence[i]]
			dataX.append([char_to_int[char] for char in char_sequence_input])
			dataY.append(char_sequence_output)

	dataX = dataX[:100]
	dataY = dataY[:100]

	# Get longest sentence and additional stats
	seq_length = 0
	for sentence in dataX:
		if seq_length < len(sentence):
			seq_length = len(sentence)

	print "Longest seq:", seq_length

	dataX = sequence.pad_sequences(dataX)

	n_patterns = len(dataX)
	print "Total Patterns: ", n_patterns
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	if train:

		# define the checkpoint
		# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
		filepath = "test.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		# fit the model
		model.fit(X, y, nb_epoch=1, batch_size=128, callbacks=callbacks_list)

	else:
		# load the network weights
		filename = "test.hdf5"
		model.load_weights(filename)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# pick a random seed
		start = numpy.random.randint(0, len(dataX) - 1)
		pattern = dataX[start]
		print "Seed:"
		print ''.join([int_to_char[value] for value in pattern])
		# generate characters
		for i in range(1000):
			x = numpy.reshape(pattern, (1, len(pattern), 1))
			x = x / float(n_vocab)
			prediction = model.predict(x, verbose=0)
			index = numpy.argmax(prediction)
			result = int_to_char[index]
			seq_in = [int_to_char[value] for value in pattern]
			if index == 0:
				print 0,
			else:
				print result,
			pattern = numpy.append(pattern, index)
			pattern = pattern[1:len(pattern)]
		print "\nDone."


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	parser.add_argument("--batch_size", type=int, default=128)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	if args.mode == "train":
		main(train=True)
	elif args.mode == "generate":
		main(train=False)

main()
