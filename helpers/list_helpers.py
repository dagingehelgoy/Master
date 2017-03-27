from __future__ import division

# Add all project modules to sys.path
import os
import sys

# Get root dir (parent of parent of main.py)
ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(__file__, os.pardir)))) + "/"
sys.path.append(ROOT_DIR)

import math
import pickle
from random import randint

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import sklearn.metrics.pairwise
from theano import tensor


import multiprocessing as mp
import time


def split_list(data, split_ratio=0.8, convert_to_np=True):
	first = data[:int((len(data) * split_ratio))]
	last = data[int((len(data) * split_ratio)):]
	if convert_to_np:
		return np.asarray(first), np.asarray(last)
	else:
		return first, last


def insert_and_remove_last(index, array, element):
	array.insert(index, element)
	del array[-1]
	return array


def tf_l2norm(tensor_array):
	norm = tf.sqrt(tf.reduce_sum(tf.pow(tensor_array, 2)))
	tensor_array /= norm
	return tensor_array


def theano_l2norm(X):
	""" Compute L2 norm, row-wise """
	norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
	X /= norm[:, None]
	return X


def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def cosine_similiary(v1, v2):
	v1 = v1.reshape(1, -1)
	v2 = v2.reshape(1, -1)
	return sklearn.metrics.pairwise.cosine_similarity(v1, v2)


def generate_sorted_similarity(image_vector_tuple):

	image_filname, image_vector, image_vector_pairs = image_vector_tuple

	total_size = len(image_vector_pairs)
	# Numver of random images to compare
	size = 100
	start = randint(0, total_size - size * 2)
	image_vector_pairs = image_vector_pairs[start:start + size]

	first_image_vector = image_vector_pairs[0][1]
	first_image_filename = image_vector_pairs[0][0]
	first_image_mse = cosine_similiary(image_vector, first_image_vector)

	best_image_vector_tuple_list = [("", 0) for i in range(size)]

	best_image_vector_tuple_list = insert_and_remove_last(0, best_image_vector_tuple_list,
	                                                      (first_image_filename, first_image_mse))

	for temp_image_name, temp_image_vector in image_vector_pairs[1:]:
		temp_image_mse = cosine_similiary(image_vector, temp_image_vector)
		for index in range(size):
			should_insert = temp_image_mse < best_image_vector_tuple_list[index][1]
			if should_insert:
				best_image_vector_tuple_list = insert_and_remove_last(index, best_image_vector_tuple_list,
				                                                      (temp_image_name, temp_image_mse))
				break

	return image_filname, best_image_vector_tuple_list


def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=30):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		barLength   - Optional  : character length of bar (Int)
	"""
	formatStr = "{0:." + str(decimals) + "f}"
	percents = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


def totuple(np_array):
	"""
	:param np_array:numpy array
	:return: tuples with tuples if np_array is multidim
	"""
	try:
		return tuple(totuple(i) for i in np_array)
	except TypeError:
		return np_array
