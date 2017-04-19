import os

ROOT_DIR = os.path.dirname((os.path.abspath(__file__))) + "/"

# vgg | inception
IMAGE_EMBEDDING_METHOD = "inception"
IMAGE_EMBEDDING_DIMENSIONS = 4096 if IMAGE_EMBEDDING_METHOD == "vgg" else 2048

# word2vec | glove | sequence
WORD_EMBEDDING_METHOD = "glove"

# Flickr8k | Flickr30k
DATASET = "Flickr8k"

RES_DIR = ROOT_DIR + "res/"
IMAGE_DIR = ROOT_DIR + "data/datasets/" + DATASET + "/images/"

DB_SUFFIX = "%s-%s-%s" % (IMAGE_EMBEDDING_METHOD, WORD_EMBEDDING_METHOD, DATASET)
DB_FILE_PATH = ROOT_DIR + "/data/database/sqlite/database-%s.db" % DB_SUFFIX

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "data/embeddings/stored-embeddings/"
STORED_EMBEDDINGS_NAME = "%s" % DB_SUFFIX

# RESULT_TEXTFILE_PATH = ROOT_DIR + "models/word2visualvec/results/results.txt"
