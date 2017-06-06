from GAN.embedding import *
from GAN.helpers.datagen import generate_index_sentences, generate_input_noise, \
	generate_string_sentences, \
	emb_generate_caption_training_batch, preprocess_sentences
from GAN.helpers.enums import WordEmbedding, Conf
from GAN.onehot import oh_create_generator, oh_create_discriminator, oh_get_training_batch

import time
import numpy as np

from data.embeddings.helpers.embeddings_helper import fetch_custom_embeddings


def generator_containing_discriminator(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model


def print_training_info(batch_counter, g_count, loss_diff, d_loss, g_loss, d_acc, g_acc):
	print "Batch: %03.0f Generator trained %s times\tdiff: % 8.7f\td_loss: % 8.7f\tg_loss: % 8.7f\td_acc: %s\tg_acc: %s" % (
		batch_counter, g_count, loss_diff, d_loss, g_loss, d_acc, g_acc)


def change_learning_rate(discriminator_on_generator):
	optimizer = Adam(lr=0.1)
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print "Changed optimizer Adam lr=0.1"


def train(gan_logger, resume_training, config):
	# if gan_logger.exists:
		# raw_input("\nModel already trained.\nPress enter to continue.\n")
	#
	print "Generating data..."
	if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		all_raw_caption_data, _, word_to_id_dict = generate_index_sentences(config, cap_data=config[Conf.DATASET_SIZE])
	else:
		# Generate image captions
		if config[Conf.IMAGE_CAPTION]:
			# filenames, all_image_vectors, captions = fetch_embeddings()
			filenames, all_image_vectors, captions = fetch_custom_embeddings(config)
			all_raw_caption_data, word_embedding_dict = preprocess_sentences(config, captions)
			del captions, filenames
		else:
			all_raw_caption_data, word_embedding_dict = generate_string_sentences(config)

	print "Compiling gan..."
	if config[Conf.IMAGE_CAPTION]:
		# g_model, d_model, gan_model = emb_create_image_gan(config)
		# g_model, d_model, gan_model = emb_create_image_gan_merge(config)
		# g_model, d_model, gan_model = emb_create_image_gan_prepend(config)
		# g_model, d_model, gan_model = emb_create_image_gan_replace_noise(config)
		g_model, d_model, gan_model = emb_gan_func_only_text(config)
		# g_model, d_model, gan_model = emb_gan_seq_only_text(config)
	elif config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		g_model = oh_create_generator(config)
		d_model = oh_create_discriminator(config)
		gan_model = generator_containing_discriminator(g_model, d_model)
	else:
		print "GAN FROM OLD"
		g_model = emb_create_generator(config)
		d_model = emb_create_discriminator(config)
		gan_model = generator_containing_discriminator(g_model, d_model)

	start_epoch = 0
	if resume_training:
		print "Resuming training..."
		g_model = load_generator(gan_logger)
		d_model = load_discriminator(gan_logger)
		g_weights = gan_logger.get_generator_weights()
		d_weights = gan_logger.get_discriminator_weights()

		print "Num g_weights: %s" % len(g_weights)
		print "Num d_weights: %s" % len(d_weights)

		largest_g_weight = 0
		largest_g_weight_size = 0

		largest_d_weight = 0
		largest_d_weight_size = 0

		for i in range(len(g_weights)):
			g_weight = g_weights[i]
			d_weight = d_weights[i]
			g_w_number = int(g_weight.split("-")[1])
			d_w_number = int(d_weight.split("-")[1])
			if g_w_number > largest_g_weight_size:
				largest_g_weight_size = g_w_number
				largest_g_weight = g_weight
			if d_w_number > largest_d_weight_size:
				largest_d_weight_size = d_w_number
				largest_d_weight = d_weight

		g_model.load_weights(
			"GAN/GAN_log/%s/model_files/stored_weights/%s" % (gan_logger.name_prefix, largest_g_weight))
		d_model.load_weights(
			"GAN/GAN_log/%s/model_files/stored_weights/%s" % (gan_logger.name_prefix, largest_d_weight))
		g_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
		d_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
		gan_model = generator_containing_discriminator(g_model, d_model)
		start_epoch = largest_g_weight_size
		print "Starting training on epoch %s" % start_epoch
	else:
		gan_logger.save_model(g_model, "generator")
		gan_logger.save_model(d_model, "discriminator")

	total_training_data = len(all_raw_caption_data)
	nb_batches = int(total_training_data / config[Conf.BATCH_SIZE])
	print("Number of batches: %s" % nb_batches)
	for epoch_cnt in range(start_epoch, config[Conf.EPOCHS], 1):
		start_time_epoch = time.time()
		print("Epoch: %s\t%s" % (epoch_cnt, gan_logger.name_prefix))

		# Shuffle data
		if config[Conf.IMAGE_CAPTION]:
			shuffle_indices = np.arange(all_raw_caption_data.shape[0])
			np.random.shuffle(shuffle_indices)
			all_raw_caption_data = all_raw_caption_data[shuffle_indices]
			all_image_vectors = all_image_vectors[shuffle_indices]
		else:
			np.random.shuffle(all_raw_caption_data)

		for batch_counter in range(nb_batches):
			# if batch_counter % 10 == 0:
			# 	print_progress(batch_counter, nb_batches, prefix="Training batches")
			raw_caption_training_batch = all_raw_caption_data[batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[Conf.BATCH_SIZE]]

			if config[Conf.IMAGE_CAPTION]:
				real_image_batch = np.asarray(all_image_vectors[
				                              batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[
					                              Conf.BATCH_SIZE]])
				# noise_image_training_batch = generate_image_with_noise_training_batch(real_image_batch, config)

			if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
				real_caption_batch = oh_get_training_batch(raw_caption_training_batch, word_to_id_dict, config)
			else:
				real_caption_batch = emb_generate_caption_training_batch(raw_caption_training_batch, word_embedding_dict, config)

			if config[Conf.IMAGE_CAPTION]:
				# fake_generated_caption_batch = g_model.predict([real_image_batch, noise_image_training_batch])
				noise_image_training_batch = generate_input_noise(config)
				A = np.repeat(noise_image_training_batch, 4, axis=0)
				B = np.reshape(A, (config[Conf.BATCH_SIZE], 4, 50))
				C = np.reshape(real_image_batch, (config[Conf.BATCH_SIZE], 1, 50))
				D = np.append(C, B, axis=1)
				fake_generated_caption_batch = g_model.predict(D)
			else:
				noise_batch = generate_input_noise(config)
				fake_generated_caption_batch = g_model.predict(noise_batch)

			# training_batch_y_zeros = np.random.uniform(0.0, 0.3, config[Conf.BATCH_SIZE])
			# training_batch_y_ones = np.random.uniform(0.7, 1.2, config[Conf.BATCH_SIZE])

			training_batch_y_zeros = np.zeros(config[Conf.BATCH_SIZE])
			training_batch_y_ones = np.ones(config[Conf.BATCH_SIZE])

			# Train discriminator

			# start_time_d = time.time()
			d_model.trainable = True
			if config[Conf.IMAGE_CAPTION]:
				# fake_images = np.random.uniform(real_image_batch.min(), real_image_batch.max(),
				#                                 size=real_image_batch.shape)
				# d_loss_fake_img, d_acc_fake_img = d_model.train_on_batch([fake_images, real_caption_batch], training_batch_y_zeros)

				d_loss_train, d_acc_train = d_model.train_on_batch([real_caption_batch], training_batch_y_ones)
				d_loss_gen, d_acc_gen = d_model.train_on_batch([fake_generated_caption_batch], training_batch_y_zeros)


				# d_loss_train, d_acc_train = d_model.train_on_batch([real_image_batch, real_caption_batch], training_batch_y_ones)
				# d_loss_gen, d_acc_gen = d_model.train_on_batch([real_image_batch, fake_generated_caption_batch], training_batch_y_zeros)
			else:
				d_loss_train, d_acc_train = d_model.train_on_batch(real_caption_batch, training_batch_y_ones)
				d_loss_gen, d_acc_gen = d_model.train_on_batch(fake_generated_caption_batch, training_batch_y_zeros)
			d_model.trainable = False
			# print("Discriminator --- %s\tseconds ---" % (time.time() - start_time_d))

			noise_batch = generate_input_noise(config)
			# start_time_g = time.time()
			# Train generator
			if config[Conf.IMAGE_CAPTION]:
				noise_image_training_batch = generate_input_noise(config)
				A = np.repeat(noise_image_training_batch, 4, axis=0)
				B = np.reshape(A, (config[Conf.BATCH_SIZE], 4, 50))
				C = np.reshape(real_image_batch, (config[Conf.BATCH_SIZE], 1, 50))
				D = np.append(C, B, axis=1)
				g_loss, g_acc = gan_model.train_on_batch(D, training_batch_y_ones)
				# g_loss, g_acc = gan_model.train_on_batch([real_image_batch, noise_image_training_batch], training_batch_y_ones)
			else:
				g_loss, g_acc = gan_model.train_on_batch(noise_batch, training_batch_y_ones)
			# print("Generator --- %s\tseconds ---" % (time.time() - start_time_g))
			if batch_counter % int(nb_batches / 1) == 0:
				print("d_loss_train:\t\t%f d_acc_train:\t\t%f" % (d_loss_train, d_acc_train))
				# if config[Conf.IMAGE_CAPTION]:
				# 	print("d_loss_fake_img:\t%f d_acc_fake_img:\t%f" % (d_loss_fake_img, d_acc_fake_img))
				print("d_loss_gen:\t\t%f d_acc_gen:\t\t%f" % (d_loss_gen, d_acc_gen))
				print("g_loss:\t\t\t%f g_acc:\t\t\t%f" % (g_loss, g_acc))
				# gan_logger.save_loss(g_loss, d_loss_gen, epoch_cnt, batch_counter)
				# if config[Conf.IMAGE_CAPTION]:
				# 	gan_logger.save_loss_acc_fake(g_loss, g_acc, d_loss_gen, d_acc_gen, d_loss_train, d_acc_train, epoch_cnt, batch_counter, d_loss_fake_img, d_acc_fake_img)
				gan_logger.save_loss_acc(g_loss, g_acc, d_loss_gen, d_acc_gen, d_loss_train, d_acc_train, epoch_cnt,
				                         batch_counter)
		# if (epoch_cnt < 1000 and epoch_cnt % 100 == 0) or (epoch_cnt < 10000 and epoch_cnt % 1000 == 0) or epoch_cnt % 5000 == 0:
		if epoch_cnt < 10 or (epoch_cnt < 100 and epoch_cnt % 10 == 0) or epoch_cnt % 100 == 0:
			gan_logger.save_model_weights(g_model, epoch_cnt, "generator")
			gan_logger.save_model_weights(d_model, epoch_cnt, "discriminator")
		print("--- %7.4f seconds ---" % (time.time() - start_time_epoch))

	gan_logger.save_model_weights(g_model, epoch_cnt, "generator")
	gan_logger.save_model_weights(d_model, epoch_cnt, "discriminator")
	print "#" * 50
	print "\tFinished with last epoch"
	print "#" * 50
