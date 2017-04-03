import datetime
import os

from GAN.helpers.enums import Conf


def generate_name_prefix(config):
	suffix = ""
	if config[Conf.NAME_SUFFIX] is not None:
		suffix = "_%s" % config[Conf.NAME_SUFFIX]
	return "%s_%s_Vocab%s_Seq%s_Batch%s_EmbSize%s_%s_Noise%s_PreInit%s_Dataset%s%s" % (
		config[Conf.DATE],
		config[Conf.WORD_EMBEDDING],
		config[Conf.VOCAB_SIZE],
		config[Conf.MAX_SEQ_LENGTH],
		config[Conf.BATCH_SIZE],
		config[Conf.EMBEDDING_SIZE],
		config[Conf.NOISE_MODE],
		config[Conf.NOISE_SIZE],
		# config[Conf.MAX_LOSS_DIFF],
		config[Conf.PREINIT],
		config[Conf.DATASET_SIZE],
		suffix
	)


class GANLogger:
	def __init__(self, config):
		self.exists = False
		self.name_prefix = generate_name_prefix(config)

		print "Initialize logging..."

		self.create_dirs("GAN/GAN_log")
		self.create_model_folders_and_files()
		self.create_model_files()

	@staticmethod
	def create_dirs(directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def create_model_files(self):
		text_filenames = ["comments", "loss", "model_summary"]
		for filename in text_filenames:
			f = open("GAN/GAN_log/%s/%s.txt" % (self.name_prefix, filename), 'a+')
			if filename == "loss":
				f.write(str(datetime.datetime.now()) + "\n")
			f.close()

	def create_model_folders_and_files(self):
		model_filepath = "GAN/GAN_log/%s/model_files/stored_weights" % self.name_prefix
		self.exists = os.path.isdir(model_filepath)
		self.create_dirs(model_filepath)

	def save_model(self, model, name):
		self.save_to_json(model, name)

	def save_to_json(self, model, name):
		model_json = model.to_json()
		with open("GAN/GAN_log/%s/model_files/%s.json" % (self.name_prefix, name), "w+") as json_file:
			json_file.write(model_json)

	def save_loss(self, g_loss, d_loss, epoch, batch):
		loss_file = open("GAN/GAN_log/%s/loss.txt" % self.name_prefix, "a")
		loss_file.write("%s,%s,%s,%s\n" % (epoch, batch, g_loss, d_loss))
		loss_file.close()

	def save_model_weights(self, model, epoch, name, suffix=""):
		path = "GAN/GAN_log/%s/model_files/stored_weights/" % self.name_prefix
		if suffix != "":
			suffix = "-" + suffix
		model.save_weights(path + "%s-%s%s" % (name, epoch, suffix), True)

	def print_start_message(self):
		print "\n"
		print "#" * 100
		print "Starting network %s" % self.name_prefix
		print "#" * 100

	def get_generator_weights(self):
		path = "GAN/GAN_log/%s/model_files/stored_weights/" % self.name_prefix
		weigthfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		generator_weights = []
		for weightfile in weigthfiles:
			if "generator" in weightfile:
				generator_weights.append(weightfile)

		return generator_weights
