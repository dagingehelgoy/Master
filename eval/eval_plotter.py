#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def distinct_number_enlarger(x):
	if x <= 3:
		return 1
	elif x <= 6:
		return 4
	elif x <= 8:
		return 7
	return 14


def plotter():
	colors = ['#F95400', '#0C56A2', '#F9DC00', '#00A670', '#C60074']
	# colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#ffffbf']

	fig = plt.figure()
	diagram = fig.add_subplot(111)
	# matplotlib.rc('font', family='Arial')

	eval_path = "/Users/markus/workspace/master/Master/eval/files/"
	# models = onehot_dropout_models()
	# models = w2v_dropout_models()
	models = w2v_hidden_models()
	axes = plt.gca()
	axes.set_ylim([0, 1])
	axes.yaxis.grid(True)
	data = []
	for model in models:
		file_path = eval_path + "beta/" + model[0] + ".txt"
		data = np.genfromtxt(
			file_path,
			delimiter=',',
			skip_header=0,
			skip_footer=0,
			names=['epoch', 'distinct_sentences', 'sentence_count', 'avg_bleu_score', 'avg_bleu_cosine',
			       'avg_bleu_tfidf', 'avg_bleu_wmd'])

		skip = 10
		itemindex = np.where(data['epoch'] == 100)[0][0]
		stop = itemindex + 170
		# stop = itemindex + 300
		# stop = itemindex + 100
		epoch_ = data['epoch'][itemindex:stop:skip]
		score_ = data['avg_bleu_score'][itemindex:stop:skip]
		distinct_sentences = [distinct_number_enlarger(x) * 5 for x in data['distinct_sentences']][::skip]
		# distinct_sentences = [x*5 for x in data['distinct_sentences']][::skip]

		color = colors.pop(0)
		diagram.plot(epoch_, score_, c=color, label=model[1])
		diagram.scatter(epoch_, score_, c=color, marker='s', s=distinct_sentences)

	# plot_all_retrival_methods(color, colors, data, diagram, distinct_sentences, epoch_, itemindex, score_, skip, stop)

	diagram.legend()
	diagram.set_xlabel('Epoch')
	# diagram.set_xlabel('Epoch', fontsize=15)
	diagram.set_ylabel(u'β')
	# diagram.set_ylabel(u'β', fontsize=15)
	plt.show()


def onehot_dropout_models():
	models = [
		(
			"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers",
			"Reference model"),
		(
			"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
			"Dropout"),
		(
			"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_softmax",
			"One-hot transformation"),
		(
			"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout-softmax",
			"Dropout & One-hot transformation"),
	]
	return models


def w2v_dropout_models(models):
	models = [
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.99dropout",
			"500-500 0.99 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
			"W2V 0.75 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.50dropout",
			"W2V 0.50 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.25dropout",
			"W2V 0.25 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.0dropout",
			"W2V 0.0 dropout"),
	]
	return models


def w2v_hidden_models():
	models = [
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.50dropout",
			"500-500"),
		(
			"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g100-d100",
			"100-100"),
		(
			"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g200-d500",
			"200-500"),
		(
			"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g500-d200",
			"500-200"),
	]
	return models


def plot_all_retrival_methods(color, colors, data, diagram, distinct_sentences, epoch_, itemindex, score_, skip, stop):
	score_cos = data['avg_bleu_cosine'][itemindex:stop:skip]
	score_tfidf = data['avg_bleu_tfidf'][itemindex:stop:skip]
	score_wmd = data['avg_bleu_wmd'][itemindex:stop:skip]
	diagram.plot(epoch_, score_, c=color, label="avg")
	diagram.scatter(epoch_, score_, c=color, marker='s', s=distinct_sentences)
	color = colors.pop(0)
	diagram.plot(epoch_, score_cos, c=color, label="cos")
	diagram.scatter(epoch_, score_cos, c=color, marker='s', s=distinct_sentences)
	color = colors.pop(0)
	diagram.plot(epoch_, score_tfidf, c=color, label="tfidf")
	diagram.scatter(epoch_, score_tfidf, c=color, marker='s', s=distinct_sentences)
	color = colors.pop(0)
	diagram.plot(epoch_, score_wmd, c=color, label="wmd")
	diagram.scatter(epoch_, score_wmd, c=color, marker='s', s=distinct_sentences)


if __name__ == '__main__':
	plotter()
