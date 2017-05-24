from GAN.config import config
from GAN.embedding import *
from GAN.helpers.enums import *
from GAN.helpers.logger import GANLogger
from GAN.onehot import oh_predict
from GAN.trainer import train


def gan_main(inference, eval, plot):
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
	elif eval:
		print "Evaluating"
		if config[Conf.WORD_EMBEDDING] == WordEmbedding.ONE_HOT:
			raise NotImplementedError
			# oh_predict(config, logger)
		else:
			if config[Conf.IMAGE_CAPTION]:
				raise NotImplementedError
			else:
				# models_to_eval = [
				# 	"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.25dropout",
				# 	"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.99dropout",
				# 	"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g100-d100",
				# ]
				# for model in models_to_eval:
				# config[Conf.MODELNAME] = model
				emb_evaluate(config, logger)
	elif plot:
		from eval.eval_plotter import plotter
		plotter(logger)
	else:
		print "training"
		train(logger, config)
