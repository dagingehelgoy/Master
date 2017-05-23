import numpy as np
from keras.engine import Input, merge, Model
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, RepeatVector
from keras.models import Sequential, model_from_json

from GAN.helpers.datagen import generate_input_noise, generate_string_sentences, generate_image_training_batch, \
	emb_generate_caption_training_batch, preprocess_sentences
from GAN.helpers.enums import Conf, PreInit
from GAN.helpers.list_helpers import *

# from data.database.helpers.pca_database_helper import fetch_pca_vector
from data.database.helpers.pca_database_helper import fetch_pca_vector
from data.embeddings.helpers.embeddings_helper import fetch_custom_embeddings
from eval.evaulator import calculate_bleu_score


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
		200,
		input_shape=(config[Conf.MAX_SEQ_LENGTH], noise_size),
		return_sequences=True)
	)
	model.add(TimeDistributed(Dense(config[Conf.EMBEDDING_SIZE], activation="tanh")))
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


def discriminator_model(config):
	model = Sequential()

	model.add(
		LSTM(
			500,
			input_shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.EMBEDDING_SIZE]),
			return_sequences=False, dropout_U=0.5, dropout_W=0.5),
	)
	model.add(Dense(1, activation="sigmoid"))
	return model


def emb_create_discriminator(config):
	d_model = discriminator_model(config)
	d_model.trainable = True
	d_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return d_model





def emb_create_image_gan(config):
	noise_size = config[Conf.IMAGE_DIM]

	# Generator

	g_lstm_input = Input(shape=(config[Conf.MAX_SEQ_LENGTH], noise_size), name="g_model_lstm_input")
	g_tensor = LSTM(500, return_sequences=True)(g_lstm_input)
	g_tensor = TimeDistributed(Dense(config[Conf.EMBEDDING_SIZE], activation='tanh'))(g_tensor)
	g_model = Model(input=g_lstm_input, output=g_tensor)

	# Discriminator

	d_lstm_input = Input(shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.EMBEDDING_SIZE]), name="d_model_lstm_input")
	d_lstm_out = LSTM(300, dropout_W=0.75, dropout_U=0.75)(d_lstm_input)

	img_input = Input(shape=(config[Conf.IMAGE_DIM],), name="d_model_img_input")
	d_tensor = merge([d_lstm_out, img_input], mode='concat')
	d_tensor = Dropout(0.75)(d_tensor)
	d_tensor = Dense(1, activation='sigmoid')(d_tensor)
	d_model = Model(input=[d_lstm_input, img_input], output=d_tensor, name="d_model")

	# GAN
	gan_tensor = d_model([g_tensor, img_input])
	gan_model = Model(input=[g_lstm_input, img_input], output=gan_tensor)

	# g_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
	d_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	gan_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

	# from keras.utils.visualize_util import plot
	# plot(g_model, to_file="g_model.png", show_shapes=True)
	# plot(d_model, to_file="d_model.png", show_shapes=True)
	# plot(gan_model, to_file="gan_model.png", show_shapes=True)
	return g_model, d_model, gan_model


def emb_create_image_gan_train_image(config):

	gan_image_input = Input(shape=(config[Conf.IMAGE_DIM],), name="gan_model_image_input")

	# Generator

	g_lstm_noise_input = Input(shape=(config[Conf.NOISE_SIZE],), name="g_model_lstm_noise_input")

	g_merge = merge([gan_image_input, g_lstm_noise_input], mode='concat')
	g_lstm_input = RepeatVector(config[Conf.MAX_SEQ_LENGTH])(g_merge)
	g_tensor = LSTM(500, return_sequences=True)(g_lstm_input)
	g_tensor = TimeDistributed(Dense(config[Conf.EMBEDDING_SIZE], activation='tanh'))(g_tensor)
	g_model = Model(input=[gan_image_input, g_lstm_noise_input], output=g_tensor)

	# Discriminator

	d_lstm_input = Input(shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.EMBEDDING_SIZE]), name="d_model_lstm_input")
	d_lstm_out = LSTM(500, dropout_W=0.5, dropout_U=0.5)(d_lstm_input)

	# img_input = Input(shape=(config[Conf.IMAGE_DIM],), name="d_model_img_input")
	d_tensor = merge([gan_image_input, d_lstm_out], mode='concat')
	d_tensor = Dropout(0.5)(d_tensor)
	d_tensor = Dense(1, activation='sigmoid')(d_tensor)
	d_model = Model(input=[gan_image_input, d_lstm_input], output=d_tensor, name="d_model")

	# GAN
	gan_tensor = d_model([gan_image_input, g_tensor])
	gan_model = Model(input=[gan_image_input, g_lstm_noise_input], output=gan_tensor)

	g_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
	d_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	gan_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

	# from keras.utils.visualize_util import plot
	# plot(g_model, to_file="g_model.png", show_shapes=True)
	# plot(d_model, to_file="d_model.png", show_shapes=True)
	# plot(gan_model, to_file="gan_model.png", show_shapes=True)
	return g_model, d_model, gan_model





def load_generator(logger):
	json_file = open("GAN/GAN_log/%s/model_files/generator.json" % logger.name_prefix, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	return model_from_json(loaded_model_json)


def load_discriminator(logger):
	json_file = open("GAN/GAN_log/%s/model_files/discriminator.json" % logger.name_prefix, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	return model_from_json(loaded_model_json)


def emb_predict(config, logger):
	print "Compiling generator..."
	noise_batch = generate_input_noise(config)
	# noise = load_pickle_file("pred.pkl")

	word_list_sentences, word_embedding_dict = generate_string_sentences(config)
	raw_caption_training_batch = word_list_sentences[np.random.randint(word_list_sentences.shape[0], size=4), :]
	# raw_caption_training_batch = np.random.choice(word_list_sentences, size=4)
	real_embedded_sentences = emb_generate_caption_training_batch(raw_caption_training_batch, word_embedding_dict, config)

	if not config[Conf.LIMITED_DATASET].endswith("_uniq.txt"):
		config[Conf.LIMITED_DATASET] = config[Conf.LIMITED_DATASET].split(".txt")[0] + "_uniq.txt"
	eval_dataset_string_list_sentences, eval_word_embedding_dict = generate_string_sentences(config)


	g_model = load_generator(logger)
	d_model = load_discriminator(logger)

	# print "Pretrained"
	# predictions = g_model.predict(noise_batch)
	# for prediction in predictions:
	# 	generated_sentence = ""
	# 	gen_most_sim_words_list = pairwise_cosine_similarity(prediction, word_embedding_dict)s
	# 	for word in gen_most_sim_words_list:
	# 		generated_sentence += word[0] + " "
	# 	print generated_sentence + "\n"

	g_weights = logger.get_generator_weights()
	d_weights = logger.get_discriminator_weights()

	print "Num g_weights: %s" % len(g_weights)
	print "Num d_weights: %s" % len(g_weights)
	# for i in range(len(g_weights)):
	for i in range(1, len(g_weights), 20):
		g_weight = g_weights[i]
		d_weight = d_weights[i]
		g_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, g_weight))
		d_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, d_weight))
		embedded_generated_sentences = g_model.predict(noise_batch[:8])
		generated_classifications = d_model.predict(embedded_generated_sentences)
		gen_header_string = "\n\nGENERATED SENTENCES: (%s)\n" % g_weight
		prediction_string = gen_header_string
		# print gen_header_string

		generated_sentences_list = []

		for j in range(len(embedded_generated_sentences)):
			embedded_generated_sentence = embedded_generated_sentences[j]
			generated_sentence = ""
			gen_most_sim_words_list = pairwise_cosine_similarity(embedded_generated_sentence, word_embedding_dict)
			for word in gen_most_sim_words_list:
				generated_sentence += word[0] + " "

			generated_sentences_list.append(generated_sentence)

			gen_sentence_string = "\n%5.4f\t%s" % (generated_classifications[j], generated_sentence)
			prediction_string += gen_sentence_string
			# print gen_sentence_string

		print "\nGenerated sentences:".upper()
		for sentence in sorted(generated_sentences_list):
			print sentence
		# print prediction_string
		calculate_bleu_score(generated_sentences_list, eval_dataset_string_list_sentences, eval_word_embedding_dict)
		print "Number of distict sentences: %s" % (len(set(generated_sentences_list)))
	# pred_header_string = "\nREAL SENTENCES: (%s)\n" % d_weight
	# prediction_string += pred_header_string
		# print pred_header_string
	# real_classifications = d_model.predict(real_embedded_sentences)
	# for j in range(len(real_classifications)):
	# 	real_sentence = ""
	# 	real_most_sim_words_list = pairwise_cosine_similarity(real_embedded_sentences[j], word_embedding_dict)
	# 	for word in real_most_sim_words_list:
	# 		real_sentence += word[0] + " "
	# 	pred_sentence_string = "\n%5.4f\t%s" % (real_classifications[j], real_sentence)
	# 	prediction_string += pred_sentence_string
	# 	print pred_sentence_string


def img_caption_predict(config, logger):
	print "Compiling generator..."
	# noise = load_pickle_file("pred.pkl")

	colors = ['black', 'blue', 'brown', 'burgundy', 'gold', 'golden', 'green', 'grey', 'indigo', 'magenta', 'orange', 'pink', 'purple', 'red', 'white', 'yellow', 'yellow-orange', 'violet']

	filenames, all_image_vectors, captions = fetch_custom_embeddings()
	all_raw_caption_data, word_embedding_dict = preprocess_sentences(config, captions)
	batch_counter = 1
	raw_caption_training_batch = all_raw_caption_data[
	                             batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[Conf.BATCH_SIZE]]

	real_caption_batch = emb_generate_caption_training_batch(raw_caption_training_batch, word_embedding_dict, config)

	# raw_caption_training_batch = word_list_sentences[np.random.randint(word_list_sentences.shape[0], size=4), :]
	# real_embedded_sentences = emb_generate_caption_training_batch(raw_caption_training_batch, word_embedding_dict, config)

	g_model = load_generator(logger)
	d_model = load_discriminator(logger)

	# print "Pretrained"
	# predictions = g_model.predict(noise_batch)
	# for prediction in predictions:
	# 	generated_sentence = ""
	# 	gen_most_sim_words_list = pairwise_cosine_similarity(prediction, word_embedding_dict)s
	# 	for word in gen_most_sim_words_list:
	# 		generated_sentence += word[0] + " "
	# 	print generated_sentence + "\n"

	g_weights = logger.get_generator_weights()
	d_weights = logger.get_discriminator_weights()

	filename_58 = 'image_02639'
	filename_65 = 'image_03182'

	pca_58 = fetch_pca_vector(filename_58)
	pca_65 = fetch_pca_vector(filename_65)

	# filename_red = 'image_02644'
	# filename_yellow = 'image_03230'
	# pca_red = fetch_pca_vector(filename_red + ".jpg")
	# pca_yellow = fetch_pca_vector(filename_red + ".jpg")
	# image_batch = np.repeat([pca_58], config[Conf.BATCH_SIZE], axis=0)
	image_batch = np.ones((config[Conf.BATCH_SIZE], config[Conf.IMAGE_DIM]))
	noise_image_training_batch = generate_input_noise(config)
	# noise_image_training_batch = generate_image_with_noise_training_batch(image_batch, config)


	print "Num g_weights: %s" % len(g_weights)
	print "Num d_weights: %s" % len(g_weights)
	prediction_string = ""
	# for i in range(len(g_weights)):
	for i in range(0, len(g_weights), 1):
	# for i in range(20, 50, 1):
		g_weight = g_weights[i]
		d_weight = d_weights[i]
		g_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, g_weight))
		d_model.load_weights("GAN/GAN_log/%s/model_files/stored_weights/%s" % (logger.name_prefix, d_weight))

		# generated_sentences = g_model.predict(noise_image_training_batch[:10])
		generated_sentences = g_model.predict([image_batch[:10], noise_image_training_batch[:10]])
		# generated_classifications = d_model.predict([image_batch, generated_sentences])

		gen_header_string = "\n\nGENERATED SENTENCES: (%s)\n" % g_weight
		prediction_string += gen_header_string
		print gen_header_string
		for j in range(len(generated_sentences)):
			embedded_generated_sentence = generated_sentences[j]
			generated_sentence = ""
			gen_most_sim_words_list = pairwise_cosine_similarity(embedded_generated_sentence, word_embedding_dict)
			for word in gen_most_sim_words_list:
				generated_sentence += word[0] + " "
			# gen_sentence_string = "\n%5.4f\t%s" % (generated_classifications[j], generated_sentence)
			gen_sentence_string = "\n%s" % generated_sentence
			prediction_string += gen_sentence_string
			print gen_sentence_string
		# print_progress(i, len(g_weights))

	# print prediction_string
	from collections import Counter
	word_count = Counter(prediction_string.split())
	tuples = []
	for color in colors:
		tuples.append((color, word_count[color]))
	tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
	for color, count in tuples:
		print "%s:\t%s" % (color, count)


def img_caption_predict_old(config, logger):
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