from enum import Enum


class NoiseMode(Enum):
	REPEAT = "repeat"
	NEW = "new"
	FIRST_EACH = "FIRST_EACH"
	FIRST_ONLY = "firstonly"
	ONES = "ones"
	ENCODING = "encoding"


class WordEmbedding(Enum):
	ONE_HOT = "onehot"
	GLOVE = "glove"
	WORD2VEC = "word2vec"


class PreInit(Enum):
	NONE = "None"
	DECODER = "Dec"
	ENCODER_DECODER = "EncDec"

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
	WORD_EMBEDDING = 10
	PREINIT = 11
	EMBEDDING_SIZE = 12
	NAME_SUFFIX = 13
	WORD2VEC_NUM_STEPS = 14
	MODELNAME = 15
