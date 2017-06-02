#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from helpers.io_helper import load_pickle_file


def hist_plotter():

	colors = ['#F95400', '#0C56A2', '#F9DC00', '#00A670', '#C60074']
	seqgan_dict = load_pickle_file("/Users/markus/workspace/master/Master/seqgan_score_dict.p")
	emb_dict = load_pickle_file("/Users/markus/workspace/master/Master/emb_score_dict.p")

	color_seqgan = colors.pop(0)
	color_emb = colors.pop(0)

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

	num_bins = 10

	fig, ax = plt.subplots()
	plt.rc('font', family='Arial')

	# the histogram of the data
	data = np.vstack([seqgan_intervals, emb_intervals]).T
	ax.hist(data, num_bins, color=[color_seqgan, color_emb], label=["Baseline", "Our model"])
	# ax.hist(data_uniq, num_bins)
	# ax.hist(emb_intervals, num_bins, normed=1)

	# add a 'best fit' line

	ax.set_xlabel(u'β')
	ax.set_ylabel('Count')

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()
	plt.legend()
	plt.show()


def plotter():
	colors = ['#F95400', '#0C56A2', '#F9DC00', '#00A670', '#C60074']
	seqgan_dict = load_pickle_file("/Users/markus/workspace/master/Master/seqgan_score_dict.p")
	emb_dict = load_pickle_file("/Users/markus/workspace/master/Master/emb_score_dict.p")

	color_seqgan = colors.pop(0)
	color_emb = colors.pop(0)

	buckets = 10

	seqgan_count = [0 for _ in range(buckets)]
	seqgan_count_uniq = [0 for _ in range(buckets)]

	emb_count = [0 for _ in range(buckets)]
	emb_count_uniq = [0 for _ in range(buckets)]

	for (n, (c, u)) in seqgan_dict.iteritems():
		seqgan_count[int(c * 10 - 1)] += u
		seqgan_count_uniq[int(c * 10 - 1)] += 1

	for (n, (c, u)) in emb_dict.iteritems():
		emb_count[int(c * 10 - 1)] += u
		emb_count_uniq[int(c * 10 - 1)] += 1

	ind = np.arange(buckets)  # the x locations for the groups
	# width = 0.35  # the width of the bars
	width = 0.49  # the width of the bars
	alpha_uniq = 0.4

	fig, ax = plt.subplots()
	# axes = plt.gca()
	# axes.set_xlim([0.5, 1.0])
	# axes.set_ylim([ymin,ymax])

	seqgan_bars = ax.bar(ind, seqgan_count, width, color=color_seqgan)
	seqgan_bars_uniq = ax.bar(ind, seqgan_count_uniq, width, color='black', alpha=alpha_uniq)
	emb_bars = ax.bar(ind + width, emb_count, width, color=color_emb)
	emb_bars_uniq = ax.bar(ind + width, emb_count_uniq, width, color='black', alpha=alpha_uniq)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Count')
	ax.set_xlabel(u'β')
	ax.set_xticks(ind + width / 2)
	x_tick_labels = []
	for i in range(buckets):
		x_tick_labels.append( "%.1f-%.1f" % ( float(i)/buckets, float(i+1)/buckets))
		# x_tick_labels.append("%.1f" % (float(i + 1) / buckets))
	ax.set_xticklabels(x_tick_labels)
	# ax.set_xlabel(x_tick_labels)
	plt.tick_params(
		axis='x',
		which='both',
		bottom='off',
	)
	ax.legend((seqgan_bars[0], emb_bars[0]), ('SeqGan', 'Word Embedding Model'))

	autolabel(seqgan_bars_uniq, ax, seqgan_count)
	autolabel(emb_bars_uniq, ax, emb_count)

	plt.show()


def autolabel(rects, ax, tot_values):
	"""
	Attach a text label above each bar displaying its height
	"""
	for i in range(len(rects)):
		rect = rects[i]
		tot_value = tot_values[i]
		if rect.get_height() == 0:
			continue
		height = rect.get_height()
		percentage = float(height) / float(tot_value) * 100
		ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%2.0f' % (percentage)  + "%", ha='center', va='bottom')
		ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%2.0f' % (percentage)  + "%", ha='center', va='bottom')





if __name__ == '__main__':
	plotter()
	# hist_plotter()
