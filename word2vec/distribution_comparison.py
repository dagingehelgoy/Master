import matplotlib
import numpy
from sklearn.preprocessing import normalize

from helpers.io_helper import load_pickle_file
from word2vec.word2vec_helpers import plot_collections
from sklearn.manifold import TSNE

matplotlib.use('Agg')


def compare_distributions():
	perplexity = 15
	data = "encoded"

	tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)

	suffix = "_lambda"
	# suffix = ""
	if data == "encoded":
		embs = load_pickle_file(
			"sequence_to_sequence/logs/NORM_S2S_2EMB_2017-04-07_VS2+1000_BS128_HD40_DHL1_ED50_SEQ5_WEMword2vec/encoded_data_lambda.pkl")
		embs = embs[:1000, 0, :]
	elif data == "word2vec":
		embs = load_pickle_file("word2vec/saved_models/word2vec_50d1000voc100001steps_embs.pkl")
		embs = embs[:1000, :]

	random_uniform = numpy.random.normal(size=embs.shape)
	# embs = normalize(embs, norm="l2")

	append = numpy.append(random_uniform, embs, axis=0)
	embs_pca = tsne.fit_transform(append)

	plot_collections([embs_pca[:1000], embs_pca[1000:]], ["gaussian-yellow", data + "-blue"], perplexity, suffix)
