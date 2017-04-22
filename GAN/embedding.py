from keras.engine import Input, merge, Model
from keras.layers import LSTM, TimeDistributed, Dense, Dropout
from keras.models import Sequential, model_from_json

from GAN.helpers.datagen import generate_input_noise, generate_string_sentences, generate_image_training_batch
from GAN.helpers.enums import Conf, PreInit
from GAN.helpers.list_helpers import *
from data.database.helpers.pca_database_helper import fetch_pca_vector


def get_decoder(config):
	if config[Conf.IMAGE_CAPTION]:
		print "TODO"
		raise NotImplementedError
	else:
		json_file = open("GAN/dec.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		decoder = model_from_json(loaded_model_json)
		decoder.load_weights("GAN/dec.hdf5")
		return decoder


def generator_model(config):
	if config[Conf.IMAGE_CAPTION]:
		noise_size = config[Conf.IMAGE_DIM]
	else:
		noise_size = config[Conf.NOISE_SIZE]
	model = Sequential()
	model.add(LSTM(
		config[Conf.EMBEDDING_SIZE],
		input_shape=(config[Conf.MAX_SEQ_LENGTH], noise_size),
		return_sequences=True))
	model.add(TimeDistributed(Dense(config[Conf.EMBEDDING_SIZE], activation="tanh")))
	return model


def discriminator_model(config):
	model = Sequential()
	model.add(LSTM(
		50,
		input_shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.EMBEDDING_SIZE]),
		return_sequences=False, dropout_U=0.4, dropout_W=0.4))
	model.add(Dense(1, activation="sigmoid"))
	return model


def emb_create_generator(config):
	if config[Conf.PREINIT] == PreInit.DECODER:
		print "Setting initial generator weights..."
		g_model = get_decoder(config)
	elif config[Conf.PREINIT] == PreInit.NONE:
		g_model = generator_model(config)
	else:
		g_model = None
	g_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

	return g_model


def emb_create_image_gan(config):
	if config[Conf.IMAGE_CAPTION]:
		noise_size = config[Conf.IMAGE_DIM]
	else:
		noise_size = config[Conf.NOISE_SIZE]

	# Generator

	g_lstm_input = Input(shape=(config[Conf.MAX_SEQ_LENGTH], noise_size), name="g_model_lstm_input")
	g_tensor = LSTM(config[Conf.EMBEDDING_SIZE], return_sequences=True)(g_lstm_input)
	g_tensor = TimeDistributed(Dense(config[Conf.EMBEDDING_SIZE], activation='tanh'))(g_tensor)
	g_model = Model(input=g_lstm_input, output=g_tensor)

	# Discriminator

	d_lstm_input = Input(shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.EMBEDDING_SIZE]), name="d_model_lstm_input")
	d_lstm_out = LSTM(50)(d_lstm_input)

	img_input = Input(shape=(config[Conf.IMAGE_DIM],), name="d_model_img_input")
	d_tensor = merge([d_lstm_out, img_input], mode='concat')
	d_tensor = Dense(1, activation='sigmoid')(d_tensor)
	d_model = Model(input=[d_lstm_input, img_input], output=d_tensor, name="d_model")

	# GAN
	gan_tensor = d_model([g_tensor, img_input])
	gan_model = Model(input=[g_lstm_input, img_input], output=gan_tensor)

	g_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
	d_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	gan_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

	# from keras.utils.visualize_util import plot
	# plot(g_model, to_file="g_model.png", show_shapes=True)
	# plot(d_model, to_file="d_model.png", show_shapes=True)
	# plot(gan_model, to_file="gan_model.png", show_shapes=True)
	return g_model, d_model, gan_model


def emb_create_discriminator(config):
	d_model = discriminator_model(config)
	d_model.trainable = True
	d_model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
	return d_model


def load_generator(logger):
	json_file = open("GAN/GAN_log/%s/model_files/generator.json" % logger.name_prefix, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	return model_from_json(loaded_model_json)


def emb_predict(config, logger):
	print "Compiling generator..."
	noise_batch = generate_input_noise(config)
	# noise = load_pickle_file("pred.pkl")

	word_list_sentences, word_embedding_dict = generate_string_sentences(config)

	g_model = load_generator(logger)

	# print "Pretrained"
	# predictions = g_model.predict(noise_batch)
	# for prediction in predictions:
	# 	sentence = ""
	# 	most_sim_words_list = pairwise_cosine_similarity(prediction, word_embedding_dict)s
	# 	for word in most_sim_words_list:
	# 		sentence += word[0] + " "
	# 	print sentence + "\n"

	weights = logger.get_generator_weights()
	print "Num weights: %s" % len(weights)
	for weight in weights:
		print "\nTesting generator: %s\n" % weight
		g_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, weight))
		predictions = g_model.predict(noise_batch[:10])

		for prediction in predictions:
			sentence = ""
			most_sim_words_list = pairwise_cosine_similarity(prediction, word_embedding_dict)
			for word in most_sim_words_list:
				sentence += word[0] + " "
			print sentence + "\n"

		# print "First sentence prediction vectors:"
		# for vector in predictions[0]:
		# 	print "%s" % (sum(vector))


def img_caption_predict(config, logger):
	g_model = load_generator(logger)
	weights = logger.get_generator_weights()

	image_to_predict = "3250589803_3f440ba781.jpg"
	pca_vector = fetch_pca_vector(image_to_predict)
	repeated_pca_vector = generate_image_training_batch([pca_vector], config)
	word_list_sentences, word_embedding_dict = generate_string_sentences(config)
	print "Num weights: %s" % len(weights)
	for weight in weights:
		print "\nTesting generator: %s\n" % weight
		g_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, weight))
		predictions = g_model.predict(repeated_pca_vector)

		for prediction in predictions:
			sentence = ""
			most_sim_words_list = pairwise_cosine_similarity(prediction, word_embedding_dict)
			for word in most_sim_words_list:
				sentence += word[0] + " "
			print sentence + "\n"
