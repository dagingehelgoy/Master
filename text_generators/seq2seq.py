import os
import warnings
from datetime import datetime

import keras
import numpy
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, RepeatVector, Embedding, TimeDistributed, Dense
from keras.models import Sequential, model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from Enums import EmbToOnehotConf, EmbeddingMethod, EmbToEmbConf
from helpers.io_helper import save_pickle_file
from helpers.list_helpers import print_progress
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from text_generators.EncoderDecoderModelCheckpoint import EncoderDecoderModelCheckpoint


def open_corpus():
	sentence_file = open("data/datasets/Flickr30k.txt")
	lines = sentence_file.readlines()
	sentence_file.close()
	return lines


def contrastive_loss(y_true, y_pred):
	shape = tf.shape(y_true) # a list: [None, 9, 2]
	dim = tf.mul(shape[1], shape[2])  # dim = prod(9,2) = 18
	y_true = tf.reshape(y_true, [-1, dim])  # -1 means "all"
	y_pred = tf.reshape(y_pred, [-1, dim])  # -1 means "all"
	x2 = tf.expand_dims(tf.transpose(y_pred, [0, 1]), 1)
	y2 = tf.expand_dims(tf.transpose(y_true, [0, 1]), 0)
	diff = y2 - x2
	maximum = tf.maximum(diff, 0.0)
	tensor_pow = tf.square(maximum)
	errors = tf.reduce_sum(tensor_pow, 2)
	diagonal = tf.diag_part(errors)
	cost_s = tf.maximum(0.05 - errors + diagonal, 0.0)
	cost_im = tf.maximum(0.05 - errors + tf.reshape(diagonal, (-1, 1)), 0.0)
	cost_tot = cost_s + cost_im
	zero_diag = tf.mul(diagonal, 0.0)
	cost_tot_diag = tf.matrix_set_diag(cost_tot, zero_diag)
	tot_sum = tf.reduce_sum(cost_tot_diag)
	return tot_sum


def embedding_to_embedding_batch_generator(data, batch_size, id_to_word, word_embeddings, seq_length, print_words=False):
	while 1:
		for pos in range(0, len(data), batch_size):
			Xs = numpy.zeros((batch_size, seq_length, 300))
			for i in range(pos, pos + batch_size):
				if print_words:
					nb_in_batch = i % batch_size
					print "Input sentence %s" % nb_in_batch
				if i < len(data):
					X = numpy.zeros((seq_length, 300))
					for j in range(len(data[i][:seq_length])):
						word = id_to_word[data[i][j]]
						if word in word_embeddings:
							if print_words:
								print word,
							X[j] = word_embeddings[word]
					Xs[i % batch_size] = X
				if print_words:
					print
			yield (Xs, Xs)


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


def get_word_embeddings():
	embeddings_index = {}
	f = open('data/embeddings/glove.6B.300d.txt')
	count = 0
	nb_close_to_zero = 0
	close_to_zero_threshold = 0.1
	for line in f:
		values = line.split()
		word = values[0]
		# distance = cosine_similarity(numpy.zeros(300), values[1:])[0][0]
		coefs = numpy.asarray(values[1:], dtype='float32')
		# if word == "biennials":
		# 	print "biennials"
		# 	for x in values[1:]:
		# 		print x,
		# 	print "Sum values: %s" % sum(float(x) for x in values[1:])
		# 	print "Sum coefs: %s" % sum(coefs)
		# 	print
		# distance = cosine_similarity(numpy.zeros(300), coefs)[0][0]
		# if distance < close_to_zero_threshold:
		# 	nb_close_to_zero += 1
		# else:
		embeddings_index[word] = coefs
		count += 1
		if count % 100 == 0:
			print_progress(count, 400000, prefix="Getting glove word embeddings")
	f.close()
	# print "Number under than %s: %s" % (close_to_zero_threshold, nb_close_to_zero)
	return embeddings_index


def get_word_embedding_matrix(word_to_id, embedding_dim):
	embeddings_index = get_word_embeddings()
	embedding_matrix = numpy.zeros((len(word_to_id) + 1, embedding_dim))
	for word, i in word_to_id.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector


def seq2seq(train=True):
	conf = EmbToEmbConf

	lines = open_corpus()

	######### Tokenizer #########
	print
	if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_ONEHOT:
		lines = ['<sos> ' + line + ' <eos>' for line in lines]
	else:
		lines = [(line.split("\t")[1]).strip() for line in lines]
	tokenizer = Tokenizer(filters="""!"#$%&'()*+-/:;=?@[\]^_`{|}~""")
	tokenizer.fit_on_texts(lines)
	sequences = tokenizer.texts_to_sequences(lines)

	word_to_id = tokenizer.word_index
	id_to_word = {token: idx for idx, token in word_to_id.items()}

	sequence_lengths = [len(x) for x in sequences]
	#MAX_SEQUENCE_LENGTH = int(numpy.average(sequence_lengths) + numpy.std(sequence_lengths))
	MAX_SEQUENCE_LENGTH = 5

	print("Sequence length: %s" % MAX_SEQUENCE_LENGTH)
	print("Nb words: %s" % conf.NB_WORDS)
	print("Batch size: %s" % conf.BATCH_SIZE)

	encoder = Sequential()
	if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_ONEHOT:
		embedding_layer = Embedding(conf.NB_WORDS + 2, conf.EMBEDDING_DIMENSION, input_length=MAX_SEQUENCE_LENGTH,
									trainable=True, mask_zero=True)
		encoder.add(embedding_layer)
	encoder.add(LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION),
					 return_sequences=False))
	encoder.add(RepeatVector(MAX_SEQUENCE_LENGTH))  # Get the last output of the RNN and repeats it

	decoder = Sequential()
	decoder.add(
		LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM), return_sequences=True))
	for _ in range(1, conf.DECODER_HIDDEN_LAYERS):
		decoder.add(LSTM(output_dim=conf.HIDDEN_DIM, return_sequences=True))

	if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_EMBEDDING:
		decoder.add(TimeDistributed(
			Dense(output_dim=conf.EMBEDDING_DIMENSION, input_shape=(MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
				  activation=conf.OUTPUT_LAYER_ACTIVATION)))
	else:
		decoder.add(TimeDistributed(Dense(output_dim=conf.NB_WORDS, input_shape=(MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
										  activation=conf.OUTPUT_LAYER_ACTIVATION)))

	model = Sequential()
	model.add(encoder)
	model.add(decoder)

	model.summary()

	if train:

		if conf.LOSS == "contrastive_loss":
			model.compile(metrics=['accuracy'], loss=contrastive_loss, optimizer='adam')
		else:
			model.compile(metrics=['accuracy'], loss=conf.LOSS, optimizer='adam')


		log_folder = "ENC_DEC_S2S_" + conf.EMBEDDING_METHOD + "_" + str(datetime.now().date()) + "_VS2+" + str(
			conf.NB_WORDS) + "_BS" + str(conf.BATCH_SIZE) + "_HD" + str(conf.HIDDEN_DIM) + "_DHL" + str(
			conf.DECODER_HIDDEN_LAYERS) + "_ED" + str(conf.EMBEDDING_DIMENSION)
		log_dir = "text_generators/logs/"
		if not os.path.exists(log_dir + log_folder):
			os.makedirs(log_dir + log_folder)
		if not os.path.exists(log_dir + log_folder + "/weights"):
			os.makedirs(log_dir + log_folder + "/weights")
		print "Working on: %s" % log_folder
		# define the checkpoint
		filepath = log_dir + log_folder + "/weights/E:{epoch:02d}-L:{loss:.4f}.hdf5"
		data = sequence.pad_sequences(sequences, MAX_SEQUENCE_LENGTH)
		numpy.random.shuffle(data)

		if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_EMBEDDING:
			word_embeddings = get_word_embeddings()
			val_gen = embedding_to_embedding_batch_generator(sequences[-conf.VAL_DATA_SIZE:], conf.BATCH_SIZE,
															 id_to_word, word_embeddings, MAX_SEQUENCE_LENGTH)
			train_gen = embedding_to_embedding_batch_generator(sequences[:-conf.VAL_DATA_SIZE], conf.BATCH_SIZE,
															   id_to_word, word_embeddings, MAX_SEQUENCE_LENGTH)
		else:
			val_gen = batch_generator(data[-conf.VAL_DATA_SIZE:], conf.BATCH_SIZE, conf.NB_WORDS)
			train_gen = batch_generator(data[:-conf.VAL_DATA_SIZE], conf.BATCH_SIZE, conf.NB_WORDS)

		if 'SSH_CONNECTION' in os.environ.keys():
			tensorboard = keras.callbacks.TensorBoard(log_dir='text_generators/logs/' + log_folder, histogram_freq=1,
													  write_graph=True)
			es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
			checkpoint = EncoderDecoderModelCheckpoint(decoder, encoder, start_after_epoch=10, filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=3)
			callbacks_list = [checkpoint, tensorboard, es]
			model.fit_generator(generator=train_gen, samples_per_epoch=len(data) - conf.VAL_DATA_SIZE,
								validation_data=val_gen, nb_val_samples=conf.VAL_DATA_SIZE, nb_epoch=conf.EPOCHS,
								callbacks=callbacks_list)

			model_json = model.to_json()
			with open(log_dir + log_folder + "/model.json", "w") as json_file:
				json_file.write(model_json)
		else:

			checkpoint = EncoderDecoderModelCheckpoint(decoder, encoder, start_after_epoch=2, filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True,
													   mode='min',
													   period=3)
			from keras.utils.visualize_util import plot
			plot(model, show_shapes=True, to_file='text_generators/logs/' + log_folder + "/model.png")
			model.fit_generator(train_gen, len(data), conf.EPOCHS, callbacks=[checkpoint])

	else:
		inference = False

		if inference:

			model_filename = "ENCODER_S2S_2EMB_2017-03-23_VS2+all_BS128_HD100_DHL1_ED300"
			weights_filename = "E:17-L:0.0782.hdf5"

			filename = "text_generators/logs/" + model_filename + "/weights/" + weights_filename
			saved_model_path = "text_generators/logs/" + model_filename + "/model.json"
			if os.path.exists(saved_model_path):
				with open(saved_model_path, "r") as json_file:
					loaded_model_json = json_file.read()
				test_model = model_from_json(loaded_model_json)
			else:
				test_model = model

			test_model.load_weights(filename)

			if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_ONEHOT:
				data = sequence.pad_sequences(sequences, MAX_SEQUENCE_LENGTH)
				numpy.random.shuffle(data)
				predict_embedding_to_onehot(conf, data, id_to_word, test_model)

			if conf.EMBEDDING_METHOD == EmbeddingMethod.EMBEDDING_TO_EMBEDDING:
				nb_predictions = 3
				numpy.random.shuffle(sequences)
				word_embeddings = get_word_embeddings()
				test_gen = embedding_to_embedding_batch_generator(sequences[-conf.VAL_DATA_SIZE:], nb_predictions, id_to_word, word_embeddings, MAX_SEQUENCE_LENGTH, print_words=True)
				test_data = test_gen.next()[0]
				preds = test_model.predict(test_data, verbose=0)
				print
				for pred in preds:
					most_sim_words_list = pairwise_cosine_similarity(pred, word_embeddings)
					sentence = ""
					for word in most_sim_words_list:
						sentence += word[0] + " "
					print sentence + "\n"
					# for vec in pred:
					# 	print "%s " % sum([x for x in vec]),
					print

		else:
			encoder_filename = "text_generators/logs/ENC_DEC_S2S_2EMB_2017-03-23_VS2+all_BS128_HD100_DHL1_ED300/weights/E:17-L:0.0792.hdf5_encoder"
			encoder.load_weights(encoder_filename)

			nb_predictions = 3
			numpy.random.shuffle(sequences)
			word_embeddings = get_word_embeddings()
			test_gen = embedding_to_embedding_batch_generator(sequences[-conf.VAL_DATA_SIZE:], nb_predictions, id_to_word,
															  word_embeddings, MAX_SEQUENCE_LENGTH)
			test_data = test_gen.next()[0]
			encoded_preds = encoder.predict(test_data, verbose=0)
			save_pickle_file(encoded_preds, "text_generators/logs/ENC_DEC_S2S_2EMB_2017-03-23_VS2+all_BS128_HD100_DHL1_ED300/prediction.pkl")


			decoder_filename = "text_generators/logs/ENC_DEC_S2S_2EMB_2017-03-23_VS2+all_BS128_HD100_DHL1_ED300/weights/E:17-L:0.0792.hdf5_decoder"
			decoder.load_weights(decoder_filename)
			decoded_preds = encoder.predict(encoded_preds, verbose=0)

			print
			for pred in decoded_preds:
				most_sim_words_list = pairwise_cosine_similarity(pred, word_embeddings)
				sentence = ""
				for word in most_sim_words_list:
					sentence += word[0] + " "
				print sentence + "\n"
				print


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


def pairwise_cosine_similarity(predicted_word_vectors, glove_dictionary):
	glove_vectors = glove_dictionary.values()
	glove_words = glove_dictionary.keys()

	glove_vectors.append(numpy.zeros(300))
	glove_words.append("0")

	cos_dis_matrix = cosine_similarity(predicted_word_vectors, glove_vectors)

	predicted_word_vector_count = len(predicted_word_vectors)

	all_word_vectors_count = len(glove_vectors)
	most_similar_words_list = []
	for predicted_image_index in range(predicted_word_vector_count):
		similarities = []
		for glove_vector_index in range(all_word_vectors_count):
			glove_word = glove_words[glove_vector_index]
			cos_sim = cos_dis_matrix[predicted_image_index][glove_vector_index]
			similarities.append((glove_word, cos_sim))
		similarities.sort(key=lambda s: s[1], reverse=True)
		most_similar_words = [x[0] for x in similarities[:5]]
		most_similar_words_list.append(most_similar_words)

	return most_similar_words_list
