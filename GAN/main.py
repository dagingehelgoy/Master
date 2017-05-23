from GAN.config import config
from GAN.embedding import *
from GAN.helpers.enums import *
from GAN.helpers.logger import GANLogger
from GAN.onehot import oh_predict
from GAN.trainer import train


def gan_main(inference, eval):
	logger = GANLogger(config, inference)
	logger.print_start_message()
	if inference:
		print "Predicting"
		if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
			oh_predict(config, logger)
		else:
			if config[Conf.IMAGE_CAPTION]:
				img_caption_predict(config, logger)
			else:
				emb_predict(config, logger)
	if eval:
		print "Evaluating"
		if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
			raise NotImplementedError
			# oh_predict(config, logger)
		else:
			if config[Conf.IMAGE_CAPTION]:
				raise NotImplementedError
			else:
				emb_evaluate(config, logger)
	else:
		print "training"
		train(logger, config)
