import os

import keras
import numpy
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, RepeatVector, Embedding, TimeDistributed, Dense
from keras.models import Sequential, model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from enums import EmbToOnehotConf
from sequence_to_sequence.embedding_seq2seq import get_word_embeddings, set_model_name
from sequence_to_sequence.encoder_decoder_model_checkpoint import EncoderDecoderModelCheckpoint


def open_corpus():
	sentence_file = open("data/datasets/Flickr30k.txt")
	lines = sentence_file.readlines()
	sentence_file.close()
	return lines


def batch_generator(data, batch_size, nb_vocab):
	while 1:
		for pos in range(0, len(data), batch_size):
			Xs = []
			Ys = []
			for i in range(pos, pos + batch_size):
				if i < len(data):
					X = data[i]
					Xs.append(X)
					Y = to_categorical(X, nb_vocab)
					Ys.append(Y)
			yield (numpy.asarray(Xs), numpy.asarray(Ys))


def get_word_embedding_matrix(word_to_id, embedding_dim, word_embedding_method):
	embeddings_index = get_word_embeddings(word_embedding_method)
	embedding_matrix = numpy.zeros((len(word_to_id) + 1, embedding_dim))
	for word, i in word_to_id.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector


def get_model(conf):
	encoder = Sequential()
	embedding_layer = Embedding(conf.NB_WORDS + 2, conf.EMBEDDING_DIMENSION, input_length=conf.MAX_SEQUENCE_LENGTH,
								trainable=True, mask_zero=True)
	encoder.add(embedding_layer)
	encoder.add(LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION),
					 return_sequences=False))
	encoder.add(RepeatVector(conf.MAX_SEQUENCE_LENGTH))  # Get the last output of the RNN and repeats it
	decoder = Sequential()
	decoder.add(
		LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
			 return_sequences=True))
	for _ in range(1, conf.DECODER_HIDDEN_LAYERS):
		decoder.add(LSTM(output_dim=conf.HIDDEN_DIM, return_sequences=True))
	decoder.add(TimeDistributed(Dense(output_dim=conf.NB_WORDS, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
									  activation=conf.OUTPUT_LAYER_ACTIVATION)))
	model = Sequential()
	model.add(encoder)
	model.add(decoder)
	model.summary()
	return decoder, encoder, model


def predict_embedding_to_onehot(conf, data, id_to_word, test_model):
	test_gen = batch_generator(data[:-conf.VAL_DATA_SIZE], conf.BATCH_SIZE, conf.NB_WORDS)
	test_data = test_gen.next()[0]
	preds = test_model.predict(test_data, verbose=0)
	for batch_index in range(len(preds)):
		print "Input sentence " + str(batch_index + 1)
		for word in test_data[batch_index]:
			if word == 0:
				print "0 ",
			else:
				print id_to_word[word] + " ",
		print
		for word in preds[batch_index]:
			argmax = numpy.argmax(word)
			if argmax == 0:
				print "0 ",
			else:
				print id_to_word[argmax] + " ",
		print


def infer(conf, id_to_word, model_filename, sequences, weights_filename):
	filename = "sequence_to_sequence/logs/" + model_filename + "/weights/" + weights_filename
	saved_model_path = "sequence_to_sequence/logs/" + model_filename + "/model.json"
	if os.path.exists(saved_model_path):
		with open(saved_model_path, "r") as json_file:
			loaded_model_json = json_file.read()
		test_model = model_from_json(loaded_model_json)
	else:
		_, _, model = get_model(conf)
		test_model = model
	test_model.load_weights(filename)
	data = sequence.pad_sequences(sequences, conf.MAX_SEQUENCE_LENGTH)
	numpy.random.shuffle(data)
	predict_embedding_to_onehot(conf, data, id_to_word, test_model)


def train_model(conf, sequences):
	decoder, encoder, model = get_model(conf)

	model.compile(metrics=['accuracy'], loss=conf.LOSS, optimizer='adam')
	filepath, log_dir, log_folder = set_model_name(conf)
	data = sequence.pad_sequences(sequences, conf.MAX_SEQUENCE_LENGTH)
	numpy.random.shuffle(data)
	val_gen = batch_generator(data[-conf.VAL_DATA_SIZE:], conf.BATCH_SIZE, conf.NB_WORDS)
	train_gen = batch_generator(data[:-conf.VAL_DATA_SIZE], conf.BATCH_SIZE, conf.NB_WORDS)
	if 'SSH_CONNECTION' in os.environ.keys():
		tensorboard = keras.callbacks.TensorBoard(log_dir='sequence_to_sequence/logs/' + log_folder, histogram_freq=1,
												  write_graph=True)
		es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
		checkpoint = EncoderDecoderModelCheckpoint(decoder, encoder, start_after_epoch=10, filepath=filepath,
												   monitor='val_loss', verbose=1, save_best_only=True, mode='min',
												   period=3)
		callbacks_list = [checkpoint, tensorboard, es]
		model.fit_generator(generator=train_gen, samples_per_epoch=len(data) - conf.VAL_DATA_SIZE,
							validation_data=val_gen, nb_val_samples=conf.VAL_DATA_SIZE, nb_epoch=conf.EPOCHS,
							callbacks=callbacks_list)

		model_json = model.to_json()
		with open(log_dir + log_folder + "/model.json", "w") as json_file:
			json_file.write(model_json)
	else:

		checkpoint = EncoderDecoderModelCheckpoint(decoder, encoder, start_after_epoch=2, filepath=filepath,
												   monitor='val_loss', verbose=1, save_best_only=True,
												   mode='min',
												   period=3)
		from keras.utils.visualize_util import plot
		plot(model, show_shapes=True, to_file='sequence_to_sequence/logs/' + log_folder + "/model.png")
		model.fit_generator(train_gen, len(data), conf.EPOCHS, callbacks=[checkpoint])


def onehot_seq2seq(inference=False, encode_data=False):
	conf = EmbToOnehotConf

	lines = open_corpus()
	lines = ['<sos> ' + line + ' <eos> <pad>' for line in lines]
	tokenizer = Tokenizer(filters="""!"#$%&'()*+-/:;=?@[\]^_`{|}~""")
	tokenizer.fit_on_texts(lines)
	sequences = tokenizer.texts_to_sequences(lines)

	word_to_id = tokenizer.word_index
	id_to_word = {token: idx for idx, token in word_to_id.items()}

	if inference:
		model_filename = "S2S_2EMB_2017-03-27_VS2+1000_BS128_HD10_DHL1_ED20_WEMword2vec"
		weights_filename = "E:131-L:0.0294.hdf5"

		infer(conf, id_to_word, model_filename, sequences, weights_filename)
	else:
		train_model(conf, sequences)
