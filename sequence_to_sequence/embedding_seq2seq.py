import os
import random

from datetime import datetime
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Lambda
from keras.models import Sequential, model_from_json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from enums import W2VEmbToEmbConf
from helpers.io_helper import load_pickle_file, save_pickle_file
from helpers.list_helpers import print_progress
from sequence_to_sequence.encoder_decoder_model_checkpoint import EncoderDecoderModelCheckpoint


def contrastive_loss(y_true, y_pred):
	shape = tf.shape(y_true)  # a list: [None, 9, 2]
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


def batch_generator(data, conf):
	while 1:
		for pos in range(0, len(data), conf.BATCH_SIZE):
			Xs = np.zeros((conf.BATCH_SIZE, conf.MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION))
			for i in range(pos, pos + conf.BATCH_SIZE):
				if i < len(data):
					X = np.zeros((conf.MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION))
					for j in range(len(data[i][:conf.MAX_SEQUENCE_LENGTH])):
						X[j] = data[i][j]
					Xs[i % conf.BATCH_SIZE] = X
			yield (Xs, Xs)


def get_word_embedding_matrix(word_to_id, embedding_dim, word_embedding_method):
	embeddings_index = get_word_embeddings(word_embedding_method)
	embedding_matrix = np.zeros((len(word_to_id) + 1, embedding_dim))
	for word, i in word_to_id.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector


def load_model(conf, model_filename, weights_filename):
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
	return test_model


def load_encoder(conf, model_filename, weights_filename):
	filename = "sequence_to_sequence/logs/" + model_filename + "/weights/" + weights_filename
	model = get_encoder(conf)
	model.load_weights(filename)
	return model


def get_random_data(conf, nb_predictions, embedded_data, string_training_data):
	random_sentences = []
	random_vectors = []
	for _ in range(nb_predictions):
		index = random.randint(0, len(embedded_data))
		random_sentences.append(string_training_data[index][:conf.MAX_SEQUENCE_LENGTH])
		random_vectors.append(embedded_data[index])
	return random_sentences, random_vectors


def get_word_embeddings(conf):
	if conf.WORD_EMBEDDING_METHOD == 'glove':
		embeddings_index = {}
		f = open('data/embeddings/glove.6B.300d.txt')
		count = 0
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			count += 1
			if count % 100 == 0:
				print_progress(count, 400000, prefix="Getting glove word embeddings")
		f.close()
		return embeddings_index

	elif conf.WORD_EMBEDDING_METHOD == 'word2vec':
		embedding_dict_name = "word2vec/saved_models/word2vec_%sd%svoc100001steps_dict_%s.pkl" % (
		conf.EMBEDDING_DIMENSION, conf.NB_WORDS, conf.DATASET if conf.DATASET != None else "flickr")
		return load_pickle_file(embedding_dict_name)

	print("WORD_EMBEDDING_METHOD not found")
	return None


def set_model_name(conf):
	log_folder = "NORM_S2S_" + conf.EMBEDDING_METHOD + "_" + str(datetime.now().date()) + "_VS2+" + str(
		conf.NB_WORDS) + "_BS" + str(conf.BATCH_SIZE) + "_HD" + str(conf.HIDDEN_DIM) + "_DHL" + str(
		conf.DECODER_HIDDEN_LAYERS) + "_ED" + str(
		conf.EMBEDDING_DIMENSION) + "_SEQ" + str(conf.MAX_SEQUENCE_LENGTH) + "_WEM" + conf.WORD_EMBEDDING_METHOD + "_DATA" + conf.DATASET
	log_dir = "sequence_to_sequence/logs/"
	if not os.path.exists(log_dir + log_folder):
		os.makedirs(log_dir + log_folder)
	else:
		raw_input("\nModel already trained.\nPress enter to continue.\n")
	if not os.path.exists(log_dir + log_folder + "/weights"):
		os.makedirs(log_dir + log_folder + "/weights")
	print "Working on: %s" % log_folder
	filepath = log_dir + log_folder + "/weights/E:{epoch:02d}-L:{val_loss:.4f}.hdf5"
	return filepath, log_dir, log_folder


def train_model(conf, data):
	decoder, encoder, model = get_model(conf)

	if conf.LOSS == "contrastive_loss":
		model.compile(metrics=['accuracy'], loss=contrastive_loss, optimizer='adam')
	else:
		model.compile(
			metrics=['accuracy', 'mean_absolute_error'],
			loss=conf.LOSS, optimizer='adam')

	filepath, log_dir, log_folder = set_model_name(conf)

	val_gen = batch_generator(data[-conf.VAL_DATA_SIZE:], conf)
	train_gen = batch_generator(data[:-conf.VAL_DATA_SIZE], conf)
	#if 'SSH_CONNECTION' in os.environ.keys():
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

	save_model(log_dir, log_folder, model, "model")
	save_model(log_dir, log_folder, encoder, "encoder")
	save_model(log_dir, log_folder, decoder, "decoder")
	#else:

		#checkpoint = EncoderDecoderModelCheckpoint(decoder, encoder, start_after_epoch=2, filepath=filepath,
												#monitor='val_loss', verbose=1, save_best_only=True,
												#mode='min',
												#period=3)
		#from keras.utils.visualize_util import plot
		#plot(model, show_shapes=True, to_file='sequence_to_sequence/logs/' + log_folder + "/model.png")
		#model.fit_generator(train_gen, len(data), conf.EPOCHS, callbacks=[checkpoint])


def save_model(log_dir, log_folder, model, name):
	model_json = model.to_json()
	with open(log_dir + log_folder + "/" + name + ".json", "w") as json_file:
		json_file.write(model_json)


def get_model(conf):
	encoder = get_encoder(conf)
	decoder = get_decoder(conf)
	model = Sequential()
	model.add(encoder)
	model.add(decoder)
	model.summary()
	return decoder, encoder, model


def get_decoder(conf):
	decoder = Sequential()
	decoder.add(LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
					 return_sequences=True))
	for _ in range(1, conf.DECODER_HIDDEN_LAYERS):
		decoder.add(LSTM(output_dim=conf.HIDDEN_DIM, return_sequences=True))
	decoder.add(TimeDistributed(
		Dense(output_dim=conf.EMBEDDING_DIMENSION, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.HIDDEN_DIM),
			  activation=conf.OUTPUT_LAYER_ACTIVATION)))
	return decoder


def get_encoder(conf):
	encoder = Sequential()
	encoder.add(LSTM(output_dim=conf.HIDDEN_DIM, input_shape=(conf.MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION),
					 return_sequences=False))
	encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
	encoder.add(RepeatVector(conf.MAX_SEQUENCE_LENGTH))  # Get the last output of the RNN and repeats it
	return encoder


def pairwise_cosine_similarity(predicted_word_vectors, glove_dictionary):
	glove_vectors = np.asarray(glove_dictionary.values())
	glove_words = glove_dictionary.keys()

	# glove_vectors.append(np.zeros(300))
	# glove_words.append("0")

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


def get_flickr_sentences():
	print "Loading Flickr sentences..."
	path = "data/datasets/Flickr30k.txt"
	sentence_file = open(path)
	word_captions = sentence_file.readlines()
	sentence_file.close()
	word_captions = [(line.split("\t")[1]).strip() for line in word_captions]
	return word_captions


def get_flowers_sentences():
	print "Loading Flower sentences..."
	path = "data/datasets/all_flowers.txt"
	sentence_file = open(path)
	word_captions = sentence_file.readlines()
	sentence_file.close()
	word_captions = [line.strip() for line in word_captions]
	return word_captions


def generate_embedding_captions(conf):
	if conf.DATASET == "flowers":
		print "A"
		sentences = get_flowers_sentences()
	else:
		sentences = get_flickr_sentences()
	word_list_sentences = []
	for sentence in sentences:
		if conf.WORD_EMBEDDING_METHOD == "word2vec":
			word_list = ["<sos>"]
		else:
			word_list = []
		for word in sentence.split(" "):
			word_list.append(word.lower())
		if conf.WORD_EMBEDDING_METHOD == "word2vec":
			word_list.append("<eos>")
		word_list_sentences.append(word_list)

	word_embedding_dict = get_word_embeddings(conf)
	return np.asarray(word_list_sentences), word_embedding_dict


def emb_get_training_batch(training_batch, word_embedding_dict, conf):
	embedding_lists = []
	for word_list in training_batch:
		embedding_sentence = []
		for word_string in word_list:
			if word_string in word_embedding_dict:
				word_embedding = word_embedding_dict[word_string]
				embedding_sentence.append(word_embedding)
		if len(embedding_sentence) > conf.MAX_SEQUENCE_LENGTH:
			embedding_sentence = embedding_sentence[:conf.MAX_SEQUENCE_LENGTH]
		while len(embedding_sentence) < conf.MAX_SEQUENCE_LENGTH:
			zeros = np.zeros(conf.EMBEDDING_DIMENSION)
			embedding_sentence.insert(0, zeros)
		embedding_lists.append(embedding_sentence)
	return np.asarray(embedding_lists)


def infer(conf, inference_sentences, inference_vectors, model_filename, weights_filename, word_embeddings):
	test_model = load_model(conf, model_filename, weights_filename)

	preds = test_model.predict(np.asarray(inference_vectors), verbose=0)
	for i in range(len(preds)):
		print "Sentence %s" % i
		most_sim_words_list = pairwise_cosine_similarity(preds[i], word_embeddings)
		# most_sim_words_list = pairwise_cosine_similarity(inference_vectors[i], word_embeddings)
		print " ".join(inference_sentences[i])
		sentence = ""
		for word in most_sim_words_list:
			sentence += word[0] + " "
		# sentence += "(" + " ".join(word) + ") "
		print sentence + "\n"


def decode(conf, string_training_data, predictions, model_filename, weights_filename, word_embeddings):
	filename = "sequence_to_sequence/logs/" + model_filename + "/weights/" + weights_filename
	decoder = get_decoder(conf)
	decoder.load_weights(filename)
	predictions = decoder.predict(predictions)
	# save_pickle_file(predictions, "decoder_predictions.pkl")

	for i in range(5):
		most_sim_words_list = pairwise_cosine_similarity(predictions[i], word_embeddings)
		print " ".join(string_training_data[i][:5])
		sentence = ""
		for word in most_sim_words_list:
			sentence += word[0] + " "
		# sentence += "(" + " ".join(word) + ") "
		print sentence + "\n"


def encode(conf, embedded_data, string_training_data, model_filename, weights_filename, word_embeddings):
	test_model = load_encoder(conf, model_filename, weights_filename + "_encoder")
	latent_data = test_model.predict(np.asarray(embedded_data))
	decode(conf, string_training_data, latent_data, model_filename, weights_filename + "_decoder",
		   word_embeddings)
	save_pickle_file(latent_data, "sequence_to_sequence/logs/" + model_filename + "/encoded_data.pkl")


def seq2seq(inference=False, encode_data=False, decode_random=False, conf=W2VEmbToEmbConf, model_filename="NORM_DROP_S2S_2EMB_2017-04-27_VS2+1000_BS128_HD500_DHL1_ED50_SEQ5_WEMword2vec", weights_filename="E:101-L:0.0104.hdf5"):
	string_training_data, word_embedding_dict = generate_embedding_captions(conf)
	embedded_data = emb_get_training_batch(string_training_data, word_embedding_dict, conf)

	if inference:
		sample_string_data, sample_embedded_data = get_random_data(conf, 10, embedded_data[-conf.VAL_DATA_SIZE:], string_training_data[-conf.VAL_DATA_SIZE:])
		infer(conf, sample_string_data, sample_embedded_data, model_filename, weights_filename, word_embedding_dict)
		# infer(conf, string_training_data[:10][:conf.MAX_SEQUENCE_LENGTH], embedded_data[:10][:conf.MAX_SEQUENCE_LENGTH], model_filename, weights_filename, word_embedding_dict)
	elif encode_data:
		encode(conf, embedded_data, string_training_data, model_filename, weights_filename, word_embedding_dict)
	elif decode_random:
		latent_data = np.random.normal(size=(conf.BATCH_SIZE, conf.MAX_SEQUENCE_LENGTH, conf.EMBEDDING_DIMENSION))
		decode(conf, string_training_data, latent_data, model_filename, weights_filename + "_decoder",
			   word_embedding_dict)
	else:
		np.random.shuffle(embedded_data)
		train_model(conf, embedded_data)
