from keras.models import Sequential
from keras.optimizers import Adam

from GAN.embedding import emb_create_generator, emb_create_discriminator, emb_create_image_gan
from GAN.helpers.datagen import generate_index_sentences, generate_input_noise, \
	generate_string_sentences, \
	emb_generate_caption_training_batch, generate_embedding_captions_from_captions, \
	generate_image_training_batch
from GAN.helpers.enums import WordEmbedding, Conf
from GAN.onehot import oh_create_generator, oh_create_discriminator, oh_get_training_batch
from data.embeddings.helpers.embeddings_helper import *


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


def train(gan_logger, config):
	# if gan_logger.exists:
	# 	raw_input("\nModel already trained.\nPress enter to continue.\n")

	print "Generating data..."
	if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
		all_raw_caption_data, _, _ = generate_index_sentences(config, cap_data=config[Conf.DATASET_SIZE])
		g_model = oh_create_generator(config)
		d_model = oh_create_discriminator(config)
	else:
		# Generate image captions
		if config[Conf.IMAGE_CAPTION]:
			filenames, all_image_vectors, captions = fetch_embeddings(10)
			all_raw_caption_data, word_embedding_dict = generate_embedding_captions_from_captions(config, captions)
			del captions, filenames
		else:
			all_raw_caption_data, word_embedding_dict = generate_string_sentences(config)

	print "Compiling gan..."
	if config[Conf.IMAGE_CAPTION]:
		g_model, d_model, gan_model = emb_create_image_gan(config)
	else:
		g_model = emb_create_generator(config)
		d_model = emb_create_discriminator(config)
		gan_model = generator_containing_discriminator(g_model, d_model)

	gan_logger.save_model(g_model, "generator")
	gan_logger.save_model(d_model, "discriminator")

	total_training_data = len(all_raw_caption_data)
	nb_batches = int(total_training_data / config[Conf.BATCH_SIZE])
	print("Number of batches: %s" % nb_batches)
	for epoch_cnt in range(config[Conf.EPOCHS]):
		print("Epoch: %s" % epoch_cnt)
		np.random.shuffle(all_raw_caption_data)
		for batch_counter in range(nb_batches):
			# if batch_counter % 10 == 0:
			# 	print_progress(batch_counter, nb_batches, prefix="Training batches")
			raw_caption_training_batch = all_raw_caption_data[batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[Conf.BATCH_SIZE]]

			if config[Conf.IMAGE_CAPTION]:
				raw_image_training_batch = np.asarray(all_image_vectors[batch_counter * config[Conf.BATCH_SIZE]:(batch_counter + 1) * config[Conf.BATCH_SIZE]])
				image_training_batch = generate_image_training_batch(raw_image_training_batch, config)

			if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
				caption_training_batch = oh_get_training_batch(raw_caption_training_batch, config)
			else:
				caption_training_batch = emb_generate_caption_training_batch(raw_caption_training_batch, word_embedding_dict, config)


			if config[Conf.IMAGE_CAPTION]:
				generated_batch = g_model.predict(image_training_batch)
			else:
				noise_batch = generate_input_noise(config)
				generated_batch = g_model.predict(noise_batch)

			training_batch_y_zeros = np.random.uniform(0.0, 0.3, config[Conf.BATCH_SIZE])
			training_batch_y_ones = np.random.uniform(0.7, 1.2, config[Conf.BATCH_SIZE])

			# Train discriminator

			d_model.trainable = True
			if config[Conf.IMAGE_CAPTION]:
				d_loss_train, d_acc_train = d_model.train_on_batch([caption_training_batch, raw_image_training_batch], training_batch_y_ones)
				d_loss_gen, d_acc_gen = d_model.train_on_batch([generated_batch, raw_image_training_batch], training_batch_y_zeros)
			else:
				d_loss_train, d_acc_train = d_model.train_on_batch(caption_training_batch, training_batch_y_ones)
				d_loss_gen, d_acc_gen = d_model.train_on_batch(generated_batch, training_batch_y_zeros)
			d_model.trainable = False

			# Train generator
			if config[Conf.IMAGE_CAPTION]:
				g_loss, g_acc = gan_model.train_on_batch([image_training_batch, raw_image_training_batch], training_batch_y_ones)
			else:
				noise_batch = generate_input_noise(config)
				g_loss, g_acc = gan_model.train_on_batch(noise_batch, training_batch_y_ones)

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
