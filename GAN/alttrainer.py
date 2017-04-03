from keras.models import Sequential
from keras.optimizers import Adam

from GAN_embedding import emb_create_generator, emb_create_discriminator
from GAN_onehot import oh_create_generator, oh_create_discriminator
from data.embeddings.helpers.embeddings_helper import *
from helpers.data_gen import generate_index_captions, generate_input_noise, to_categorical_lists, onehot_to_softmax, \
	generate_embedding_captions_from_flickr30k, emb_get_training_batch
from helpers.enums import WordEmbedding, Conf


def generator_containing_discriminator(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model


def train(gan_logger, config):
	if gan_logger.exists:
		raw_input("\nModel already trained.\nPress enter to continue.\n")

	print "Generating data..."
	if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		training_data, _, _ = generate_index_captions(config, cap_data=config[Conf.DATASET_SIZE])
	else:
		# filenames, image_vectors, captions = fetch_embeddings(10)
		# caption_training_data, word_embedding_dict = generate_embedding_captions_from_captions(config, captions)
		# print caption_training_data
		training_data, word_embedding_dict = generate_embedding_captions_from_flickr30k(config)

	print "Compiling generator..."
	if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		g_model = oh_create_generator(config, preinit=config[Conf.PREINIT])
	else:
		g_model = emb_create_generator(config, preinit=config[Conf.PREINIT])

	print "Compiling discriminator..."
	if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		d_model = oh_create_discriminator(config)
	else:
		d_model = emb_create_discriminator(config)

	print "Compiling gan..."
	discriminator_on_generator = generator_containing_discriminator(g_model, d_model)

	gan_logger.save_model(g_model, "generator")
	gan_logger.save_model(d_model, "discriminator")

	total_training_data = len(training_data)
	nb_batches = int(total_training_data / config[Conf.BATCH_SIZE])
	g_loss_list = [0.0 for _ in range(50)]
	print("Number of batches: %s" % nb_batches)
	for epoch_cnt in range(config[Conf.EPOCHS]):
		print("Epoch: %s" % epoch_cnt)
		np.random.shuffle(training_data)
		for batch_counter in range(nb_batches):
			noise_batch = generate_input_noise(config)
			training_batch = training_data[
			                 batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[Conf.BATCH_SIZE]]

			if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
				training_batch = oh_get_training_batch(training_batch, config)
			else:
				training_batch = emb_get_training_batch(training_batch, word_embedding_dict, config)

			generated_batch = g_model.predict(noise_batch)

			training_batch_x = np.concatenate((training_batch, generated_batch))
			# if config[Conf.WORD_REPR] == WordRepr.ONE_HOT:
			# training_batch_y_zeros = [0] * config[Conf.BATCH_SIZE]
			# training_batch_y_ones = [1] * config[Conf.BATCH_SIZE]
			training_batch_y_zeros = np.random.uniform(0.0, 0.3, config[Conf.BATCH_SIZE])
			training_batch_y_ones = np.random.uniform(0.7, 1.2, config[Conf.BATCH_SIZE])
			training_batch_y = training_batch_y_ones + training_batch_y_zeros
			# else:
			# 	training_batch_y_zeros = np.zeros((config[Conf.BATCH_SIZE], config[Conf.MAX_SEQ_LENGTH], 1))
			# 	training_batch_y_ones = np.ones((config[Conf.BATCH_SIZE], config[Conf.MAX_SEQ_LENGTH], 1))
			# 	training_batch_y = np.concatenate((training_batch_y_ones, training_batch_y_zeros))

			d_model.trainable = True
			# Train discriminator
			# d_loss_gen, d_acc_gen = d_model.train_on_batch(training_batch_x, training_batch_y)
			d_loss_train, d_acc_train = d_model.train_on_batch(training_batch, training_batch_y_ones)
			d_loss_gen, d_acc_gen = d_model.train_on_batch(generated_batch, training_batch_y_zeros)
			d_model.trainable = False

			# Train generator
			noise_batch = generate_input_noise(config)
			# training_batch_y_ones = np.asarray(training_batch_y_ones)
			# training_batch_y_zeros = np.asarray(training_batch_y_zeros)
			g_loss, g_acc = discriminator_on_generator.train_on_batch(noise_batch, training_batch_y_ones)

			# g_loss_list.append(g_loss)
			# g_loss_list.pop(0)

			# loss_diff = g_loss - d_loss_gen
			# g_count = 1
			# changed_learning_rate = False
			# while loss_diff > config[Conf.MAX_LOSS_DIFF] and g_count < 10000:
			# noise_batch = generate_input_noise(config)
			# while loss_diff > config[Conf.MAX_LOSS_DIFF]:
			# for _ in range(1):
			# if g_count > 500 and not changed_learning_rate:
			# 	change_learning_rate(discriminator_on_generator)
			# 	changed_learning_rate = True

			# if d_acc_gen < 1:
			# 	print "DISC ACC: %s" % d_acc_gen
			# g_loss, g_acc = discriminator_on_generator.train_on_batch(noise_batch, training_batch_y_ones)

			# generated_batch = g_model.predict(noise_batch)
			# d_loss_gen, d_acc_gen = d_model.test_on_batch(generated_batch, training_batch_y_zeros)

			# g_loss_list.append(g_loss)
			# g_loss_list.pop(0)
			# if g_loss_list[0] == g_loss_list[-1] and g_count > 5000:
			# 	print "#" * 50
			# 	print "\tSTOPPED AND SAVED LOCAL MINIMA"
			# 	print "#" * 50
			# 	gan_logger.save_model_weights(g_model, epoch_cnt, batch_counter, "generator",
			# 	                              suffix="LOCAL-MINIMA")
			# 	gan_logger.save_model_weights(d_model, epoch_cnt, batch_counter, "discriminator",
			# 	                              suffix="LOCAL-MINIMA")
			# 	return

			# loss_diff = g_loss - d_loss_gen
			# g_count += 1

			# if g_count % 1000 == 0:
			# 	print_training_info(batch_counter, g_count, loss_diff, d_loss_gen, g_loss, d_acc_gen, g_acc)

			# if changed_learning_rate:
			# 	if config[Conf.WORD_REPR] == WordRepr.ONE_HOT:
			# 		discriminator_on_generator.compile(loss='binary_crossentropy', optimizer="adam",
			# 		                                   metrics=['accuracy'])
			# 	elif config[Conf.WORD_REPR] == WordRepr.EMBEDDING:
			# 		discriminator_on_generator.compile(loss='binary_crossentropy', optimizer="adam",
			# 		                                   metrics=['accuracy'])

			# if g_count > 1000:
			# 	print_training_info(batch_counter, g_count, loss_diff, d_loss_gen, g_loss, d_acc_gen, g_acc)

			# d_model.trainable = True

			if batch_counter % int(nb_batches / 1) == 0:
				print("d_loss_gen:\t%f d_acc_gen:\t%f" % (d_loss_gen, d_acc_gen))
				print("d_loss_train:\t%f d_acc_train:\t%f" % (d_loss_train, d_acc_train))
				print("g_loss:\t\t%f g_acc:\t\t%f" % (g_loss, g_acc))
				gan_logger.save_loss(g_loss, d_loss_gen, epoch_cnt, batch_counter)

		if (epoch_cnt < 100 and epoch_cnt % 10 == 0) or (
						epoch_cnt < 1000 and epoch_cnt % 100 == 0) or epoch_cnt % 500 == 0:
			gan_logger.save_model_weights(g_model, epoch_cnt, "generator")
			gan_logger.save_model_weights(d_model, epoch_cnt, "discriminator")
		if g_loss and d_loss_gen and batch_counter:
			gan_logger.save_loss(g_loss, d_loss_gen, epoch_cnt, batch_counter)

	gan_logger.save_model_weights(g_model, epoch_cnt, "generator")
	gan_logger.save_model_weights(d_model, epoch_cnt, "discriminator")
	print "#" * 50
	print "\tFinished with last epoch"
	print "#" * 50


def print_training_info(batch_counter, g_count, loss_diff, d_loss, g_loss, d_acc, g_acc):
	print "Batch: %03.0f Generator trained %s times\tdiff: % 8.7f\td_loss: % 8.7f\tg_loss: % 8.7f\td_acc: %s\tg_acc: %s" % (
		batch_counter, g_count, loss_diff, d_loss, g_loss, d_acc, g_acc)


def oh_get_training_batch(batch, config):
	tr_one_hot_caption_batch = to_categorical_lists(batch, config)

	tr_softmax_caption_batch = onehot_to_softmax(tr_one_hot_caption_batch)
	return tr_softmax_caption_batch


def change_learning_rate(discriminator_on_generator):
	optimizer = Adam(lr=0.1)
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print "Changed optimizer Adam lr=0.1"
