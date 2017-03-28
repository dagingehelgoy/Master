from enum import Enum


class NoiseMode(Enum):
	REPEAT = "repeat"
	NEW = "new"
	FIRST_EACH = "FIRST_EACH"
	FIRST_ONLY = "firstonly"
	ONES = "ones"
	ENCODING = "encoding"


class WordRepr(Enum):
	ONE_HOT = "onehot"
	EMBEDDING = "embedding"


class WordEmbedding(Enum):
	GLOVE = 1
	Word2Vec20d1000 = "word2vec20d1000.pkl"


class Conf(Enum):
	DATE = 1
	VOCAB_SIZE = 2
	MAX_SEQ_LENGTH = 3
	BATCH_SIZE = 4
	DATASET_SIZE = 5
	NOISE_SIZE = 6
	NOISE_MODE = 7
	MAX_LOSS_DIFF = 8
	EPOCHS = 9
	WORD_REPR = 10
	LOAD_GENERATOR = 11
	EMBEDDING_SIZE = 12
	WORD_EMBEDDING = 13
	NAME_SUFFIX = 14


class ConfigClass():
	VOCAB_SIZE = 5000
	BATCH_SIZE = 100


if __name__ == '__main__':
	confClass = ConfigClass()
	confClass.VOCAB_SIZE
