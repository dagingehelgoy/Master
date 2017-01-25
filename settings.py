import os
import sys

ROOT_DIR = os.path.dirname((os.path.abspath(__file__))) + "/"

# vgg | inception
IMAGE_EMBEDDING_METHOD = "vgg"
IMAGE_EMBEDDING_DIMENSIONS = 4096 if IMAGE_EMBEDDING_METHOD == "vgg" else 2048


# word2vec | glove | sequence
WORD_EMBEDDING_METHOD = "glove"

# Flickr8k | Flickr30k
DATASET = "Flickr30k"

RES_DIR = ROOT_DIR + "res/"
IMAGE_DIR = ROOT_DIR + "data/datasets/" + DATASET + "/images/"
CREATE_NEGATIVE_EXAMPLES = False

DB_SUFFIX = "%s-%s-%s" % (IMAGE_EMBEDDING_METHOD, WORD_EMBEDDING_METHOD, DATASET)
DB_FILE_PATH = ROOT_DIR + "/data/database/sqlite/database-%s.db" % DB_SUFFIX

# Word2Vec
WORD_EMBEDDING_DIMENSION = 300
WORD_EMBEDDING_DIR = ROOT_DIR + "models/word2vec/embeddings/"

if DATASET == "Flickr8k":
	WORD_FILEPATH = ROOT_DIR + "data/datasets/Flickr8k/Flickr8k.token.txt"
else:
	WORD_FILEPATH = ROOT_DIR + "data/datasets/Flickr30k/flickr30k/results_20130124.token"

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "data/embeddings/stored-embeddings/"
NEG_TAG = "neg" if CREATE_NEGATIVE_EXAMPLES else ""
STORED_EMBEDDINGS_NAME = "%s-%s" % (DB_SUFFIX, NEG_TAG)

RESULT_TEXTFILE_PATH = ROOT_DIR + "models/word2visualvec/results/results.txt"


