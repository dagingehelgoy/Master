import keras.backend as k
import numpy as np
import tensorflow as tf


def contrastive_loss_old(_, predict):
	s, im = tf.split(predict, 2, 1)
	s2 = tf.expand_dims(tf.transpose(s, [0, 1]), 1)
	im2 = tf.expand_dims(tf.transpose(im, [0, 1]), 0)
	diff = im2 - s2
	maximum = tf.maximum(diff, 0.0)
	tensor_pow = tf.square(maximum)
	errors = tf.reduce_sum(tensor_pow, 2)
	diagonal = tf.diag_part(errors)
	cost_s = tf.maximum(0.05 - errors + diagonal, 0.0)
	cost_im = tf.maximum(0.05 - errors + tf.reshape(diagonal, (-1, 1)), 0.0)
	cost_tot = cost_s + cost_im
	zero_diag = tf.multiply(diagonal, 0.0)
	cost_tot_diag = tf.matrix_set_diag(cost_tot, zero_diag)
	tot_sum = tf.reduce_sum(cost_tot_diag)
	return tot_sum


def contrastive_loss(y_true, y_pred):
	x2 = tf.expand_dims(tf.transpose(y_pred, [0, 1]), 1)
	y2 = tf.expand_dims(tf.transpose(y_true, [0, 1]), 0)
	diff = y2 - x2
	maximum = tf.maximum(diff, 0.0)
	tensor_pow = tf.square(maximum)
	errors = tf.reduce_sum(tensor_pow, 2)
	diagonal = tf.diag_part(errors)
	cost_s = tf.maximum(0.05 - errors + diagonal, 0.0)
	cost_im = tf.maximum(0.05 - errors + tf.reshape(diagonal, (-1, 1)), 0.0)
	cost_tot = cost_s + cost_im
	zero_diag = tf.multiply(diagonal, 0.0)
	cost_tot_diag = tf.matrix_set_diag(cost_tot, zero_diag)
	tot_sum = tf.reduce_sum(cost_tot_diag)
	return tot_sum


if __name__ == '__main__':
	a_concat = [.1, .1, .1, .1]
	b_concat = [.1, .1, .1, .1]
	# c_concat = [.9, .10, .11, .12]
	dummy = [[0.0 for x in range(len(a_concat))]]

	old = np.asarray([a_concat, b_concat])
	loss_keras_old = contrastive_loss_old(dummy, tf.constant(old))
	print("Old loss: " + str(k.eval(loss_keras_old)))

	a_1 = [.1, .1]
	a_2 = [.0, .0]

	b_1 = [.1, .1]
	b_2 = [.0, .0]

	# c_1 = [.9, .10]
	# c_2 = [.11, .12]

	x = np.asarray([np.asarray(a_1), np.asarray(b_1)])
	y = np.asarray([np.asarray(a_2), np.asarray(b_2)])

	loss_keras = contrastive_loss(y, x)

	print("New loss: " + str(k.eval(loss_keras)))

	x = tf.zeros((2, 2))
	y = tf.ones((2, 2))

	loss_keras = contrastive_loss(y, x)

	print("New loss: " + str(k.eval(loss_keras)))
