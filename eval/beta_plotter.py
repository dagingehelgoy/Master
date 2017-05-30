#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from helpers.io_helper import load_pickle_file
import matplotlib.mlab as mlab

def plotter():

	colors = ['#F95400', '#0C56A2', '#F9DC00', '#00A670', '#C60074']
	seqgan_dict = load_pickle_file("/Users/markus/workspace/master/Master/seqgan_score_dict.p")
	emb_dict = load_pickle_file("/Users/markus/workspace/master/Master/emb_score_dict.p")

	seqgan_intervals = []
	seqgan_intervals_uniq = []
	for (s, (b, n)) in seqgan_dict.iteritems():
		seqgan_intervals_uniq.append(b)
		for _ in range(n):
			seqgan_intervals.append(b)

	emb_intervals = []
	emb_intervals_uniq = []
	for (s, (b, n)) in emb_dict.iteritems():
		emb_intervals_uniq.append(b)
		for _ in range(n):
			emb_intervals.append(b)

	num_bins = 20

	fig, ax = plt.subplots()
	plt.rc('font', family='Arial')
	# plt.style.use('seaborn-deep')
	#
	# x = np.random.normal(1, 2, 5000)
	# y = np.random.normal(-1, 3, 5000)
	# data = np.vstack([x, y]).T
	# bins = np.linspace(-10, 10, 30)
	#
	# plt.hist(data, bins, alpha=0.7, label=['x', 'y'])
	# plt.legend(loc='upper right')
	# plt.show()


	# the histogram of the data
	# ax.hist(seqgan_intervals, num_bins, normed=1)
	data = np.vstack([seqgan_intervals, emb_intervals]).T
	# data_uniq = np.vstack([seqgan_intervals_uniq, emb_intervals_uniq]).T
	ax.hist(data, num_bins, color=[colors.pop(0), colors.pop(0)], label=["Baseline", "Our model"])
	# ax.hist(data_uniq, num_bins)
	# ax.hist(emb_intervals, num_bins, normed=1)

	# add a 'best fit' line

	ax.set_xlabel(u'β')
	ax.set_ylabel('Count')

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()
	plt.legend()
	plt.show()



if __name__ == '__main__':
	plotter()
