from keras.layers import LSTM, TimeDistributed, Dense
from keras.models import Sequential, model_from_json
import settings
from GAN.helpers.datagen import generate_input_noise, generate_string_sentences
from GAN.helpers.enums import Conf, PreInit

# from helpers.list_helpers import *
from sequence_to_sequence.seq2seq import pairwise_cosine_similarity


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
		noise_size = settings.IMAGE_EMBEDDING_DIMENSIONS
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
		return_sequences=False))
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
