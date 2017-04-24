import os

from GAN.config import config
from GAN.helpers.enums import Conf


def fetch_flower_captions():
	# textfile = open("/Users/markus/workspace/master/Master/data/datasets/flowers/text/all.txt", 'r')
	# from collections import Counter
	# wordcount = Counter(textfile.read().split())
	# textfile.close()
	# for item in wordcount.items():
	# 	print("{}\t{}".format(*item))
	# print

	dir_path = "data/datasets/%s/text" % config[Conf.LIMITED_DATASET]
	textfile_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
	all_sentences_file = open("data/datasets/%s/text/all.txt" % config[Conf.LIMITED_DATASET], 'a')
	for textfile_dir in textfile_dirs:
		textfile_dir = os.path.join(dir_path, textfile_dir)
		textfiles = [f for f in os.listdir(textfile_dir) if os.path.isfile(os.path.join(textfile_dir, f))]
		class_sentnece_file = open("%s/class.txt" % textfile_dir, 'a')
		for textfile in textfiles:
			textfile_file = open("%s/%s" % (textfile_dir, textfile), 'r')
			textfile_sentences = [x.replace(".", "") for x in textfile_file.readlines()]
			textfile_file.close()
			all_sentences_file.writelines(textfile_sentences)
			class_sentnece_file.writelines(textfile_sentences)
		class_sentnece_file.close()
	all_sentences_file.close()


if __name__ == '__main__':
	fetch_flower_captions()
