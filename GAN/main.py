from GAN.config import config
from GAN.embedding import *
from GAN.helpers.enums import *
from GAN.helpers.logger import GANLogger
from GAN.onehot import oh_predict, oh_evaluate
from GAN.trainer import train
from GAN.config import config


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
			models_to_eval = [
					"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers",
					"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
					"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout-softmax",
					"2017-05-13_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_softmax",
				]
			for model in models_to_eval:
				config[Conf.MODELNAME] = model
				logger = GANLogger(config, inference)
				logger.print_start_message()
				oh_evaluate(config, logger)
		else:
			if config[Conf.IMAGE_CAPTION]:
				raise NotImplementedError
			else:
				models_to_eval = [
					"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.0dropout",
					"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.25dropout",
					"2017-05-10_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.50dropout",
					"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout",
					"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g100-d100",
					"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g200-d500",
					"2017-05-18_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_g500-d200",
					"2017-05-16_ImgCapFalse_word2vec_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.99dropout",
				]
				for model in models_to_eval:
					config[Conf.MODELNAME] = model
					logger = GANLogger(config, inference)
					logger.print_start_message()
					emb_evaluate(config, logger)
	elif plot:
		from eval.eval_plotter import plotter
		plotter(logger)
	else:
		print "training"
		train(logger, config)
