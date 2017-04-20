from GAN.config import config
from GAN.embedding import *
from GAN.helpers.enums import *
from GAN.helpers.logger import GANLogger
from GAN.onehot import oh_predict
from GAN.trainer import train


def gan_main(inference):
	logger = GANLogger(config)
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
	else:
		print "training"
		train(logger, config)
