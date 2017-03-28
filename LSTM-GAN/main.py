import datetime

from embedding_gan import emb_predict
from enums import NoiseMode, Conf, WordRepr, WordEmbedding
from logger import GANLogger
from lstm_gan import train
from onehot_gan import oh_predict, oh_test_generator
from keras.backend.tensorflow_backend import get_session, set_session
import tensorflow as tf
import sys

config = {
	Conf.WORD_REPR: WordRepr.EMBEDDING,

	# Conf.DATE: datetime.datetime.now().date(),
	Conf.DATE: "2017-03-27",

	# Conf.NAME_SUFFIX: "sos",
	Conf.NAME_SUFFIX: "sos_shuffle",
	# Conf.NAME_SUFFIX: None,

	Conf.VOCAB_SIZE: 1000,
	Conf.MAX_SEQ_LENGTH: 5,
	Conf.EMBEDDING_SIZE: 20,

	Conf.DATASET_SIZE: 10000,
	Conf.BATCH_SIZE: 100,
	Conf.EPOCHS: 10000,

	Conf.NOISE_SIZE: 10,
	Conf.NOISE_MODE: NoiseMode.REPEAT,

	Conf.MAX_LOSS_DIFF: 0,

	Conf.LOAD_GENERATOR: False,

	Conf.WORD_EMBEDDING: WordEmbedding.Word2Vec20d1000

}

if __name__ == '__main__':
	logger = GANLogger(config)
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	session = tf.Session(config=tf_config)
	set_session(session)
	if "train" in sys.argv:
		train(logger, config)
	elif "pred" in sys.argv:
		if config[Conf.WORD_REPR] == WordRepr.ONE_HOT:
			oh_predict(config, logger)
		else:
			emb_predict(config, logger)




# test_discriminator()
