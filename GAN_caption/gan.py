import numpy as np
from keras.engine import Input
from keras.layers import Dense, merge
from keras.models import Model
from keras.optimizers import SGD

# SETTINGS
# from keras.utils.visualize_util import plot
from data.embeddings.helpers.embeddings_helper import fetch_embeddings
from helpers.io_helper import load_pickle_file
from data.database.helpers.image_database_helper import *
from data.database.helpers.caption_database_helper import *

NOISE_DIM = 100
IMAGE_EMD_DIM = 4096
CAP_EMB_DIM = 300


def fetch_training_data():
	return [], []


def generator_model():
	g_input = Input(shape=(NOISE_DIM + IMAGE_EMD_DIM,), name='g_input')
	g_tensor = Dense(2048, activation='tanh')(g_input)
	g_tensor = Dense(1024, activation='tanh')(g_tensor)
	g_tensor = Dense(512, activation='tanh')(g_tensor)
	g_tensor = Dense(CAP_EMB_DIM, activation='tanh')(g_tensor)
	g_model = Model(input=g_input, output=g_tensor, name="generator_model")
	return g_model, g_input


def discriminator_model():
	d_input_cap = Input(shape=(CAP_EMB_DIM,), name="d_input_cap")
	d_tensor = Dense(100, activation='tanh')(d_input_cap)

	d_input_img = Input(shape=(IMAGE_EMD_DIM,), name="d_input_img")

	d_merge = merge([d_tensor, d_input_img], mode='concat')

	d_output = Dense(1, activation='sigmoid')(d_merge)
	d_model = Model(input=[d_input_cap, d_input_img], output=d_output, name="discriminator_model")

	return d_model, d_input_img


def generator_containing_discriminator(g_model, d_model, g_input, d_input_img):
	d_model.trainable = False

	gan_input = [g_model.input, d_input_img]
	gan_output = d_model([g_model.output, d_input_img])
	gan = Model(gan_input, gan_output, name="gan_model")
	return gan


def fetch_text_caption(cap_emb):
	return find_n_most_similar_captions(cap_emb, 3)


def fetch_image_filename(img_emb):
	return fetch_filename_from_image_vector(img_emb)


def train(BATCH_SIZE):
	caption_vectors, image_vectors, _ = fetch_embeddings()

	caption_vectors = np.asarray(caption_vectors)
	image_vectors = np.asarray(image_vectors)

	# save_pickle_file(caption_vectors, "test_cap.pickle")
	# save_pickle_file(image_vectors, "test_img.pickle")
	#
	# caption_vectors = load_pickle_file("test_cap.pickle")
	# image_vectors = load_pickle_file("test_img.pickle")

	print("Building models")
	d_model, discriminator_on_generator, g_model = get_models()

	d_model.trainable = True

	noise_and_img = np.zeros((BATCH_SIZE, NOISE_DIM + IMAGE_EMD_DIM))

	for epoch in range(100):
		print("Epoch is", epoch)
		training_data_count = image_vectors.shape[0]
		total_batches_count = int(training_data_count / BATCH_SIZE)

		print("Number of batches", total_batches_count)

		should_test_result = True

		for batch_index in range(total_batches_count):
			for i in range(BATCH_SIZE):
				uniform_rand = np.random.uniform(-1, 1, 100)

				consecutive_img = image_vectors[batch_index * BATCH_SIZE + i]

				noise_and_img[i, :100] = uniform_rand
				noise_and_img[i, 100:] = consecutive_img

			real_caption_batch = caption_vectors[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
			real_image_batch = image_vectors[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]

			generated_captions = g_model.predict(noise_and_img, verbose=0)

			if should_test_result and epoch != 0 and epoch % 10 == 0:
				image_filename = fetch_image_filename(real_image_batch[0])

				print "Fetching fetching most similar caption"
				caption_string = fetch_text_caption(generated_captions[0])

				result_string = "Best caption for image: %s\n%s" % (image_filename, caption_string)
				res_file = open("result.txt", 'a')
				res_file.write(result_string + "\n")
				res_file.close()
				print result_string
				should_test_result = False


			captions = np.concatenate((real_caption_batch, generated_captions))
			imgs = np.concatenate((real_image_batch, real_image_batch))
			X = [captions, imgs]
			y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

			# Train d_model
			d_loss = d_model.train_on_batch(X, y)

			for i in range(BATCH_SIZE):
				noise_and_img[i, :100] = np.random.uniform(-1, 1, 100)

			# Train g_model
			d_model.trainable = False
			g_loss = discriminator_on_generator.train_on_batch([noise_and_img, real_image_batch], [1] * BATCH_SIZE)
			d_model.trainable = True
			if batch_index % 100 == 0:
				print("batch %d d_loss : %f" % (batch_index, d_loss))
				print("batch %d g_loss : %f" % (batch_index, g_loss))

		# g_model.save_weights('g_model-%s' % epoch, True)
		# d_model.save_weights('d_model-%s' % epoch, True)
		# g_model.save_weights('g_model', True)
		# d_model.save_weights('d_model', True)


def get_models():
	d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	g_model, g_input = generator_model()
	g_model.compile(loss='binary_crossentropy', optimizer="SGD")
	d_model, d_input_img = discriminator_model()
	d_model.compile(loss='binary_crossentropy', optimizer=d_optim)
	# plot(g_model, to_file="generatorCAP.png", show_shapes=True)
	# plot(d_model, to_file="discriminatorCAP.png", show_shapes=True)
	discriminator_on_generator = generator_containing_discriminator(g_model, d_model, g_input, d_input_img)
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
	# plot(discriminator_on_generator, to_file="discriminator_on_generatorCAP.png", show_shapes=True)
	return d_model, discriminator_on_generator, g_model


def gan_main():
	train(128)


if __name__ == '__main__':
	train(1)
