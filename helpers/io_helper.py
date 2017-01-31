import os
import pickle
import settings


def save_pickle_file(data, name):
	print("Saving pickle file: %s" % name)
	f = open(name, "wb")
	pickle.dump(data, f, protocol=2)
	f.close()


def load_pickle_file(name):
	print("Loading pickle file: %s" % name)
	f = open(name, "rb")
	data = pickle.load(f)
	f.close()
	return data


def check_pickle_file(name):
	if os.path.isfile(name):
		return True
	return False


def create_missing_folders():
	directories = ["data/databases/sqlite",
	               "data/embeddings/stored-embeddings",
	               "models/word2vec/embeddings",
	               "models/word2visualvec/results",
	               "models/word2visualvec/stored_models",
	               "models/word2visualvec/model_embeddings"]

	for directory in directories:
		complete_path = settings.ROOT_DIR + directory
		if not os.path.exists(complete_path):
			os.makedirs(complete_path)

