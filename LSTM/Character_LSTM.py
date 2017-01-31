''''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

import itertools

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
np.seterr(divide="ignore")

sentence_file = open("Flickr30k.token.txt")
lines = sentence_file.readlines()
sentence_file.close()
# Put sentences in list
text_sentences = []
for line in lines:
	text_sentence = []
	for x in (((line.split(".jpg#")[1])[1:]).strip()):
		text_sentence.append(x.lower())
	text_sentences.append(text_sentence)

# text_sentences = text_sentences[:10]
print('corpus length:', sum(len(x) for x in text_sentences))
all_chars_iterator = itertools.chain.from_iterable(text_sentences)
chars = sorted(list(set(all_chars_iterator)))

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
sentences = []
next_chars = []
for text_sentence in text_sentences:
	for i in range(0, len(text_sentence) - maxlen):
		sentences.append(text_sentence[i: i + maxlen])
		next_chars.append(text_sentence[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')

	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)



train = False
if train:
	# train the model, output generated text after each iteration
	for iteration in range(1, 60):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		filepath = "weights-improvement-" + str(iteration) + "-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		model.fit(X, y, batch_size=256, nb_epoch=1, callbacks=callbacks_list)

else:
	# load the network weights
	filename = "weights-improvement-59-0.7997.hdf5"
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')



	input_sample = True

	if input_sample:
		while True:
			start_sentence = raw_input("Start sentence... ")
			start_sentence = start_sentence.lower()

			start_sentence = start_sentence[:20]
			while len(start_sentence) < 20:
				start_sentence = " " + start_sentence


			print('----- Seed: \t\t' + start_sentence)
			for diversity in [0.2, 0.5, 1.0, 1.2]:

				generated = ''
				sentence = ''.join(start_sentence)
				generated += sentence
				output = '----- diversity:' + str(diversity) + "\t" + generated
				sys.stdout.write(output)
				for i in range(100):
					x = np.zeros((1, maxlen, len(chars)))
					for t, char in enumerate(sentence):
						x[0, t, char_indices[char]] = 1.

					preds = model.predict(x, verbose=0)[0]
					next_index = sample(preds, diversity)
					next_char = indices_char[next_index]


					generated += next_char
					sentence = sentence[1:] + next_char

					sys.stdout.write(next_char)
					sys.stdout.flush()

					if next_char == '.':
						break

				print()

	else:
		for nb_examples in range(5):
			if nb_examples == 0:
				start_sentence = [' '] * maxlen
			else:
				start_sentence = sentences[random.randint(0, len(sentences))]

			for diversity in [0.2, 0.5, 1.0, 1.2]:
				print()
				print('----- diversity:', diversity)

				generated = ''
				sentence = ''.join(start_sentence)
				generated += sentence
				print('----- Generating with seed: "' + sentence + '"')
				sys.stdout.write(generated)

				for i in range(100):
					x = np.zeros((1, maxlen, len(chars)))
					for t, char in enumerate(sentence):
						x[0, t, char_indices[char]] = 1.

					preds = model.predict(x, verbose=0)[0]
					next_index = sample(preds, diversity)
					next_char = indices_char[next_index]

					generated += next_char
					sentence = sentence[1:] + next_char

					sys.stdout.write(next_char)
					sys.stdout.flush()

					if next_char == '.':
						break







print()
