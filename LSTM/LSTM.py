import itertools

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

import sys

sentence_file = open("captions.token")
lines = sentence_file.readlines()
sentence_file.close()


# Put sentences in list
sentences = []
for line in lines:
	sentence = []
	for x in ((((line.split(".jpg#")[1])[1:]).strip()).split()):
		sentence.append(x.lower())

	sentences.append(sentence)

# Give all words an unique id (int), make dictionary with both 'word' : id and id : 'word'
all_tokens = itertools.chain.from_iterable(sentences)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
id_to_word = {token: idx for idx, token in word_to_id.items()}

# Add sequences of ids corresponding to the sentences of words
index_sentences = []
for sentence in sentences:
	index_sentence = []
	for word in sentence:
		if word in word_to_id:
			index_sentence.append(word_to_id[word])
	index_sentences.append(index_sentence)

# Get longest sentence and additional stats
n_vocab = len(word_to_id)
n_words = 0
seq_length = 0
for sentence in index_sentences:
	n_words += len(sentence)
	if seq_length < len(sentence):
		seq_length = len(sentence)

# Pad sentences with zero to set length of all sentences to the length of the longest
pad_index_sentences = sequence.pad_sequences(index_sentences)

# Create training data for each word in a sentence as X and all prior words as Y for the whole sentence
dataX = []
dataY = []
for sentence in pad_index_sentences:
	for i in range(1, seq_length):
		x = numpy.zeros(seq_length)
		for value in sentence[0:i]:
			x[i] = value
		dataX.append(x)
		dataY.append(sentence[i])

dataX = numpy.asarray(dataX)

n_patterns = len(dataX)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=20, batch_size=128, callbacks=callbacks_list)
