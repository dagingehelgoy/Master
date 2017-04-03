from keras.layers import LSTM, TimeDistributed, Dense
from keras.models import Sequential

from GAN.helpers.datagen import *
from helpers.enums import Conf


def get_decoder(config):
	nb_words = 1000
	hidden_dim = 1024
	decoder_hidden_layers = 1

	decoder = Sequential()
	decoder.add(LSTM(output_dim=hidden_dim,
	                 input_shape=(20, hidden_dim),
	                 return_sequences=True))
	for _ in range(1, decoder_hidden_layers):
		decoder.add(LSTM(output_dim=hidden_dim, return_sequences=True))

	decoder.add(TimeDistributed(Dense(output_dim=nb_words, input_shape=(20, hidden_dim), activation='softmax')))

	decoder.load_weights("300_emb_decoder.hdf5")

	return decoder


def generator_model(config):
	model = Sequential()
	model.add(LSTM(
		output_dim=config[Conf.NOISE_SIZE],
		input_shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.NOISE_SIZE]),
		return_sequences=True))

	model.add(TimeDistributed(Dense(config[Conf.VOCAB_SIZE], activation="softmax")))
	return model


def discriminator_model(config):
	model = Sequential()
	model.add(LSTM(
		256,
		input_shape=(config[Conf.MAX_SEQ_LENGTH], config[Conf.VOCAB_SIZE]),
		return_sequences=False))

	model.add(Dense(1, activation="sigmoid"))
	return model


def oh_test_discriminator():
	print "Generating data..."
	index_captions, id_to_word_dict, word_to_id_dict = generate_index_captions(MAX_SEQUENCE_LENGTH, VOCAB_SIZE,
	                                                                           cap_data=DATASET_SIZE)

	# test_caption = "<sos> boy swimming in water <eos>"
	# test_caption = "<sos> <sos> <sos> <sos> <sos> <sos> <sos> <sos> <sos> <sos> <sos>"
	# print test_caption
	# test_captiontest_caption_one_hot = to_categorical_lists([test_caption_index], MAX_SEQUENCE_LENGTH, NB_WORDS)
	# print test_caption_one_hot


	# test_caption_index = []
	# for word in test_caption.split(" "):
	# 	test_caption_index.append(word_to_id_dict[word])
	# print test_caption_index


	print "Compiling generator..."
	g_model = generator_model()
	g_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
	g_model.load_weights('stored_models/2017-02-23_5000-20-256-first-10--1_g_model-60')
	print g_model.metrics_names

	print "Compiling discriminator..."
	d_model = discriminator_model()
	d_model.trainable = True
	d_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
	d_model.load_weights('stored_models/2017-02-23_5000-20-256-first-10--1_d_model-60')
	print d_model.metrics_names

	BATCH_SIZE = 10
	g_input_noise_batch = generate_input_noise(
		BATCH_SIZE,
		noise_mode=NOISE_MODE,
		max_seq_lenth=MAX_SEQUENCE_LENGTH,
		noise_size=NOISE_SIZE)
	index = 0
	index_caption_batch = index_captions[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
	one_hot_caption_batch = to_categorical_lists(index_caption_batch, config)
	generated_captions_batch = g_model.predict(g_input_noise_batch)

	# Train discriminator
	d_loss_pos = d_model.train_on_batch([one_hot_caption_batch], [1] * BATCH_SIZE)
	d_loss_neg = d_model.train_on_batch([generated_captions_batch], [0] * BATCH_SIZE)

	print "pos: %s" % d_loss_pos
	print "neg: %s" % d_loss_neg
	print "Training discriminator"
	print d_model.predict([one_hot_caption_batch])
	print d_model.predict([generated_captions_batch])


def oh_test_generator(config):
	index = 0

	print "Generating data..."
	index_captions, id_to_word_dict, word_to_id_dict = generate_index_captions(config)
	index_caption_batch = index_captions[index * config[Conf.BATCH_SIZE]:(index + 1) * config[Conf.BATCH_SIZE]]
	one_hot_caption_batch = to_categorical_lists(index_caption_batch, config)
	softmax_caption = onehot_to_softmax(one_hot_caption_batch)

	g_model = get_decoder(config)
	# g_model.compile(loss='categorical_crossentropy', optimizer="adam")

	print "Setting initial generator weights..."
	np.random.seed(42)
	g_input_noise = generate_input_noise(config)

	predictions = g_model.predict(g_input_noise)

	soft_max_vals = []
	soft_min_vals = []
	pred_max_vals = []
	pred_min_vals = []

	for i in range(10):
		soft_max_vals.append(max(softmax_caption[0][i]))
		soft_min_vals.append(min(softmax_caption[0][i]))
		pred_max_vals.append(max(predictions[0][i]))
		pred_min_vals.append(min(predictions[0][i]))

	max_softs = ""
	for val in soft_max_vals:
		max_softs += "%10.9f\t" % val
	max_preds = ""
	for val in pred_max_vals:
		max_preds += "%10.9f\t" % val
	min_softs = ""
	for val in soft_min_vals:
		min_softs += "%g\t" % val
	min_preds = ""
	for val in pred_min_vals:
		min_preds += "%g\t" % val

	print ""
	print "Max soft:\t%s" % max_softs
	print "Max pred:\t%s" % max_preds
	print ""
	print "Min: soft:\t%s" % min_softs
	print "Min: pred:\t%s" % min_preds
	print ""


def oh_create_generator(config, preinit=False):
	if preinit:
		print "Setting initial generator weights..."
		g_model = get_decoder(config)
	else:
		g_model = generator_model(config)
	g_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

	return g_model


def oh_create_discriminator(config):
	d_model = discriminator_model(config)
	d_model.trainable = True
	d_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	return d_model


def oh_predict(config, logger):
	print "Compiling generator..."

	# weights_folder = 'log/%s/model_files/stored_weights/' % logger.name_prefix
	# weights_file = "generator-0-46-LOCAL-MINIMA"

	g_model = get_decoder(config)
	# g_model.compile(loss='categorical_crossentropy', optimizer="adam")
	# g_model.load_weights(str(weights_folder) + str(weights_file))

	g_input_noise = generate_input_noise(config)

	predictions = g_model.predict(g_input_noise)

	index_captions, id_to_word_dict, word_to_id_dict = generate_index_captions(config,
	                                                                           cap_data=config[Conf.DATASET_SIZE])
	for prediction in predictions:
		sentence = ""
		for softmax_word in prediction:
			id = np.argmax(softmax_word)
			if id == 0:
				sentence += "0 "
			else:
				word = id_to_word_dict[id]
				sentence += word + " "
		print sentence + "\n"


def oh_get_training_batch(batch, config):
	tr_one_hot_caption_batch = to_categorical_lists(batch, config)

	tr_softmax_caption_batch = onehot_to_softmax(tr_one_hot_caption_batch)
	return tr_softmax_caption_batch
