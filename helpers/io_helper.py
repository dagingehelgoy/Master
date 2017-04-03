import os
import pickle
import settings


def save_pickle_file(data, name):
	f = open(name, "wb+")
	pickle.dump(data, f)
	f.close()
	print("Saved pickle file: %s" % name)


def load_pickle_file(name):
	# print("Loading pickle file: %s" % name)
	f = open(name, "rb")
	data = pickle.load(f)
	f.close()
	return data


def check_pickle_file(name):
	if os.path.isfile(name):
		return True
	return False

