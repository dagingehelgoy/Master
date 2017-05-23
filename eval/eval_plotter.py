import matplotlib.pyplot as plt
import random
import matplotlib
from GAN.config import config
import numpy as np


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
	if x <= 5:
		return 1
	elif x <= 8:
		return 5
	return 10

def plotter(logger):
	eval_file_path = logger.get_eval_file_path()
	data_1 = np.genfromtxt(
		eval_file_path,
		delimiter=',',
		skip_header=0,
		skip_footer=0,
		names=['epoch', 'distinct_sentences', 'sentence_count', 'avg_bleu_score', 'avg_bleu_cosine', 'avg_bleu_tfidf', 'avg_bleu_wmd'])

	skip = 5
	epoch_ = data_1['epoch'][::skip]
	score_ = data_1['avg_bleu_score'][::skip]
	distinct_sentences = [distinct_number_enlarger(x)*5 for x in data_1['distinct_sentences']][::skip]
	# distinct_sentences = [x*5 for x in data_1['distinct_sentences']][::skip]


	plt.plot(epoch_, score_)

	plt.scatter(epoch_, score_, marker='s', s=distinct_sentences, label='the data')
	plt.show()

