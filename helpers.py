import os
import pickle
import sys


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


def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=30):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		barLength   - Optional  : character length of bar (Int)
	"""
	formatStr = "{0:." + str(decimals) + "f}"
	percents = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()
