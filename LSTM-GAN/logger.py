import datetime
import os

from enums import Conf


def generate_name_prefix(config):
	suffix = ""
	if config[Conf.NAME_SUFFIX] is not None:
		suffix = "_%s" % config[Conf.NAME_SUFFIX]
	return "%s_%s_%s_VS%s_MS%s_BS%s_NM%s_NS%s_LD%s_DS%s%s" % (
		config[Conf.WORD_REPR],
		config[Conf.WORD_EMBEDDING],
		config[Conf.DATE],
		config[Conf.VOCAB_SIZE],
		config[Conf.MAX_SEQ_LENGTH],
		config[Conf.BATCH_SIZE],
		config[Conf.NOISE_MODE],
		config[Conf.NOISE_SIZE],
		config[Conf.MAX_LOSS_DIFF],
		config[Conf.DATASET_SIZE],
		suffix
	)


class GANLogger:
	def __init__(self, config):

		self.name_prefix = generate_name_prefix(config)

		print "\n"
		print "#" * 40
		print "Starting network %s" % self.name_prefix
		print "#" * 40

		print "Initialize logging..."

		self.create_dirs("log")
		self.create_model_folders_and_files()
		self.create_model_files()

	@staticmethod
	def create_dirs(directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def create_model_files(self):
		text_filenames = ["comments", "loss", "model_summary"]
		for filename in text_filenames:
			f = open("log/%s/%s.txt" % (self.name_prefix, filename), 'a+')
			if filename == "loss":
				f.write(str(datetime.datetime.now()) + "\n")
			f.close()

	def create_model_folders_and_files(self):
		self.create_dirs("log/%s/model_files/stored_weights" % self.name_prefix)

	def save_model(self, model, name):
		self.save_to_json(model, name)

	def save_to_json(self, model, name):
		model_json = model.to_json()
		with open("log/%s/model_files/%s.json" % (self.name_prefix, name), "w+") as json_file:
			json_file.write(model_json)

	def save_loss(self, g_loss, d_loss, epoch, batch):
		loss_file = open("log/%s/loss.txt" % self.name_prefix, "a")
		loss_file.write("%s,%s,%s,%s\n" % (epoch, batch, g_loss, d_loss))
		loss_file.close()

	def save_model_weights(self, model, epoch, name, suffix=""):
		path = "log/%s/model_files/stored_weights/" % self.name_prefix
		if suffix != "":
			suffix = "-" + suffix
		model.save_weights(path + "%s-%s%s" % (name, epoch, suffix), True)

	def get_generator_weights(self):
		path = "log/%s/model_files/stored_weights/" % self.name_prefix
		weigthfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		generator_weights = []
		for weightfile in weigthfiles:
			if "generator" in weightfile:
				generator_weights.append(weightfile)

		return generator_weights
