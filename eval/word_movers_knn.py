import numpy as np
from pyemd import emd
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances

from GAN.config import config
from GAN.helpers.enums import Conf
from helpers.io_helper import load_pickle_file
from word2vec.word2vec_helpers import get_dict_filename




def get_wmd_distance(d1, d2, word_embedding_dict, min_vocab=4, verbose=False):
	model_vocab = word_embedding_dict.keys()
	vocabulary = [w for w in set(d1.lower().split() + d2.lower().split()) if w in model_vocab and w not in stop_words.ENGLISH_STOP_WORDS]
	if len(vocabulary) < min_vocab:
		return 1
	vect = CountVectorizer(vocabulary=vocabulary).fit([d1, d2])
	feature_names = vect.get_feature_names()
	W_ = np.array([word_embedding_dict[w] for w in feature_names if w in word_embedding_dict])
	D_ = euclidean_distances(W_)
	D_ = D_.astype(np.double)
	D_ /= D_.max()  # just for comparison purposes
	v_1, v_2 = vect.transform([d1, d2])
	v_1 = v_1.toarray().ravel()
	v_2 = v_2.toarray().ravel()
	# pyemd needs double precision input
	v_1 = v_1.astype(np.double)
	v_2 = v_2.astype(np.double)
	v_1 /= v_1.sum()
	v_2 /= v_2.sum()
	if verbose:
		print vocabulary
		print v_1, v_2
	return emd(v_1, v_2, D_)


# q = "Government speaks to the media in Illinois"
q = "The flower is red with yellow stamen"
d1 = "Plant with reddish color and orange stem"
d2 = "Plant with greenish color and black stem"

# d2 = "Obama speaks to the media in Illinois"
# d3 = "The President addresses the press in Chicago"
print get_wmd_distance(q, d1, word_embedding_dict)
print get_wmd_distance(q, d2, word_embedding_dict)
