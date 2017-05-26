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
	eval_path = "/Users/markus/workspace/master/Master/eval/files/"
	models = get_onehot_models()[3:4]
	# models = get_wordemb_models(models)


	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()

	ax1.set_xlabel('Epoch')
	ax1.set_ylabel(u'β score', color=colors[0])
	ax2.set_ylabel('Generator loss', color=colors[1])

	ax1.set_ylim([0, 1])
	ax2.set_ylim([0, 17])

	x_min = 0
	x_max = 150
	ax1.set_xlim([x_min, x_max])

	for model in models:
		beta_file_path = eval_path + "beta/" + model[0] + ".txt"
		beta_data = np.genfromtxt(
			beta_file_path,
			delimiter=',',
			skip_header=0,
			skip_footer=0,
			names=['epoch', 'distinct_sentences', 'sentence_count', 'avg_bleu_score', 'avg_bleu_cosine',
			       'avg_bleu_tfidf', 'avg_bleu_wmd'])

		loss_file_path = eval_path + "loss/" + model[0] + ".txt"
		loss_data = np.genfromtxt(
			loss_file_path,
			delimiter=',',
			skip_header=1,
			skip_footer=3,
			names=['epoch', 'batch', 'loss_g', 'g_acc', 'd_loss_gen', 'd_acc_gen', 'd_loss_train', 'd_acc_train'])

		loss_d_train = loss_data["d_loss_train"]
		loss_d_gen = loss_data["d_loss_gen"]
		loss_d = (loss_d_gen + loss_d_train) / 2

		beta_skip = 5
		beta_start_index = np.where(beta_data['epoch'] == 0)[0][0]
		# beta_stop_index = np.where(beta_data['epoch'] == 99)[0][0]
		beta_stop_index = None

		loss_skip = 1
		loss_start_index = np.where(loss_data['epoch'] == 0)[0][0]
		# loss_stop_index = np.where(loss_data['epoch'] == 300)[0][0]
		loss_stop_index = None

		beta_epochs = beta_data['epoch'][beta_start_index:beta_stop_index:beta_skip]
		beta = beta_data['avg_bleu_score'][beta_start_index:beta_stop_index:beta_skip]

		# beta_epochs = np.insert(beta_epochs, 0, beta_data['epoch'][1])
		# beta = np.insert(beta, 0, beta_data['avg_bleu_score'][1])

		loss_epochs = loss_data['epoch'][loss_start_index:loss_stop_index:loss_skip]
		loss_g = loss_data['loss_g'][loss_start_index:loss_stop_index:loss_skip]
		loss_d = loss_d[loss_start_index:loss_stop_index:loss_skip]



		distinct_sentences = [distinct_number_enlarger(x) * 5 for x in beta_data['distinct_sentences']][::beta_skip]
		# distinct_sentences = [x*5 for x in beta_data['distinct_sentences']][::beta_skip]

		ax1.plot(beta_epochs, beta, c=colors[0])
		ax1.scatter(beta_epochs, beta, c=colors[0], marker='s', s=distinct_sentences)
		# ax2.plot(loss_epochs, loss_d, c=colors[3], label="Discriminator loss")
		ax2.plot(loss_epochs, loss_g, c=colors[1])
		plt.axvline(80, c='black', linestyle=':', label="Best observed sentences")

	ax1.legend()
	ax2.legend()
	plt.legend()
	# diagram.set_xlabel('Epoch')
	# diagram.set_xlabel('Epoch', fontsize=15)
	# diagram.set_ylabel(u'β')
	# diagram.set_ylabel(u'β', fontsize=15)
	plt.show()


def get_onehot_models():
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


def get_wordemb_models(models):
	models = [
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.99dropout",
			"500-500 0.99 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
			"500-500 0.75 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.50dropout",
			"500-500 0.50 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.25dropout",
			"500-500 0.25 dropout"),
		(
			"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.0dropout",
			"500-500 0.0 dropout"),
		(
			"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g100-d100",
			"100-100 0.50 dropout"),
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
