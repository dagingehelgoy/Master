#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# x = [x for x in range(11)]
# y = [0, 1, 2, 3, 4, 5, 4, 2, 5, 3, 2]
# s = [random.randint(1,10) for _ in range(9)]
# s.append(10)
# s.append(1)
# s = [a*5 for a in s]
# plt.errorbar(x, y, yerr=s, fmt='-o', markersize=1, color='k', label='size 2')
# plt.plot(x, y)
# plt.scatter(x, y, marker='s', s=s, label='the data')
#
# plt.show()

def distinct_number_enlarger(x):
	if x <= 4:
		return 1
	elif x <= 7:
		return 5
	return 10


def plotter():

	fig = plt.figure()
	diagram = fig.add_subplot(111)
	matplotlib.rc('font', family='Arial')

	eval_path = "/Users/markus/workspace/master/Master/eval/files/"
	# models = [
	# 	"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout-softmax",
	# 	"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
	# 	"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_softmax",
	# 	"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers",
	# ]
	models = [
		("2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.25dropout", "500-500 0.25 dropout"),
		("2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.99dropout", "500-500 0.99 dropout"),
		("2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.50dropout", "500-500 0.50 dropout"),
	]

	data = []
	for model in models:
		file_path = eval_path + model[0] + ".txt"
		data = np.genfromtxt(
			file_path,
			delimiter=',',
			skip_header=0,
			skip_footer=0,
			names=['epoch', 'distinct_sentences', 'sentence_count', 'avg_bleu_score', 'avg_bleu_cosine', 'avg_bleu_tfidf', 'avg_bleu_wmd'])

		skip = 10
		epoch_ = data['epoch'][::skip]
		score_ = data['avg_bleu_score'][::skip]
		distinct_sentences = [distinct_number_enlarger(x) * 5 for x in data['distinct_sentences']][::skip]
		# distinct_sentences = [x*5 for x in data['distinct_sentences']][::skip]

		diagram.plot(epoch_, score_)
		diagram.scatter(epoch_, score_, marker='s', s=distinct_sentences, label=model[1])
		# diagram.scatter(epoch_, score_, marker='s', s=distinct_sentences, label="One-hot " + model[114:])

	diagram.legend()
	diagram.set_xlabel('Epoch', fontsize=15)
	diagram.set_ylabel(u'β', fontsize=15)
	plt.show()


if __name__ == '__main__':
	plotter()
