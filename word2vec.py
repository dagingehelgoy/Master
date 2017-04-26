# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from GAN.config import *
from word2vec.word2vec_helpers import build_dataset, generate_batch, plot_with_labels, save_model, load_model, \
	plot_with_labels_selected

# Step 2: Build the dictionary and replace rare words with UNK token.
VOCABULARY_SIZE = config[Conf.VOCAB_SIZE]
EMBEDDING_SIZE = config[Conf.EMBEDDING_SIZE]  # Dimension of the embedding vector.
NUM_STEPS = config[Conf.WORD2VEC_NUM_STEPS]
DATASET = "flickr"  # flowers/flickr

if "plot" in sys.argv:
	reverse_dictionary, final_embeddings, _ = load_model(EMBEDDING_SIZE, VOCABULARY_SIZE, NUM_STEPS, DATASET)
	plot_with_labels(reverse_dictionary, final_embeddings,
	                 filename="word2vec_%sd%svoc%ssteps_plot_%s" % (EMBEDDING_SIZE, VOCABULARY_SIZE, NUM_STEPS, DATASET), plot_only=500)
elif "plot_selection" in sys.argv:
	reverse_dictionary, final_embeddings, dictionary = load_model(EMBEDDING_SIZE, VOCABULARY_SIZE, NUM_STEPS, DATASET)
	selected_word_list = ["man", "woman", "boy", "girl", "blue", "yellow", "green", "red", "one", "two", "three", "chair", "table", "sweater", "dress", "suit"]
	plot_with_labels_selected(reverse_dictionary, final_embeddings, selected_word_list,
	                 filename="word2vec_selection_%sd%svoc%ssteps_plot_%s" % (EMBEDDING_SIZE, VOCABULARY_SIZE, NUM_STEPS, DATASET))

else:

	# num_steps = 1

	data, count, dictionary, reverse_dictionary = build_dataset(VOCABULARY_SIZE, DATASET)
	print('Most common words (+UNK)', count[:5])
	print('Sample datasets', data[:10], [reverse_dictionary[i] for i in data[:10]])
	data_index = 0
	# Step 3: Function to generate a training batch for the skip-gram model.
	batch, labels, data_index = generate_batch(data, batch_size=8, num_skips=2, skip_window=1, data_index=data_index)

	for i in range(8):
		print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

	# Step 4: Build and train a skip-gram model.

	batch_size = 128
	skip_window = 1  # How many words to consider left and right.
	num_skips = 2  # How many times to reuse an input to generate a label.

	# We pick a random validation set to sample nearest neighbors. Here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent.
	valid_size = 16  # Random set of words to evaluate similarity on.
	valid_window = 100  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 64  # Number of negative examples to sample.

	graph = tf.Graph()

	with graph.as_default():
		# Input datasets.
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Ops and variables pinned to the CPU because of missing GPU implementation
		with tf.device('/cpu:0'):
			# Look up embeddings for inputs.
			embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			# Construct the variables for the NCE loss
			nce_weights = tf.Variable(
				tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],
				                    stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
			nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, VOCABULARY_SIZE))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Add variable initializer.
		init = tf.initialize_all_variables()

	# Step 5: Begin training.
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	with tf.Session(graph=graph, config=tf_config) as session:
		# We must initialize all variables before we use them.
		init.run()
		print("Initialized")

		average_loss = 0
		for step in range(NUM_STEPS):
			batch_inputs, batch_labels, data_index = generate_batch(data, batch_size, num_skips, skip_window,
			                                                        data_index)
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += loss_val

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss at step ", step, ": ", average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 10000 == 0:
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = "Nearest to %s:" % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = "%s %s," % (log_str, close_word)
					print(log_str)

		final_embeddings = normalized_embeddings.eval()

	plotting = False
	saving = True

	if saving:
		save_model(reverse_dictionary, final_embeddings, EMBEDDING_SIZE, VOCABULARY_SIZE, NUM_STEPS, DATASET)

	if plotting:
		plot_with_labels(reverse_dictionary, final_embeddings)
