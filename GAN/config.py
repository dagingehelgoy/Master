# coding=utf-8
import datetime

from GAN.helpers.enums import NoiseMode, Conf, WordEmbedding, PreInit

# noinspection SpellCheckingInspection
"""
	Eksperimenter som må kjøres:
		- Sammenligne noise size                (DONE) 10 vs. 50 vs. 1000
		- Teste alle NoiseModes
		- Teste VocabSize
		- Teste Embedding Size
"""

config = {
	Conf.WORD_EMBEDDING: WordEmbedding.WORD2VEC,

	Conf.DATE: datetime.datetime.now().date(),

	Conf.NAME_SUFFIX: None,
	# Conf.NAME_SUFFIX: "largeG",

	Conf.VOCAB_SIZE: 1000,
	Conf.MAX_SEQ_LENGTH: 10,

	Conf.LIMITED_DATASET: "surf.txt",
	# Conf.LIMITED_DATASET: None,
	Conf.DATASET_SIZE: -1,
	Conf.BATCH_SIZE: 10,
	Conf.EPOCHS: 100000,

	Conf.NOISE_MODE: NoiseMode.REPEAT,

	Conf.MAX_LOSS_DIFF: 0,

	Conf.EMBEDDING_SIZE: 50,
	Conf.NOISE_SIZE: 40,
	Conf.PREINIT: PreInit.NONE,

	Conf.WORD2VEC_NUM_STEPS: 100001,

	# Conf.MODELNAME: "2017-03-31_word2vec_Vocab1000_Seq5_Batch256_EmbSize50_repeat_Noise200_Dataset-1_largeDandG",
	Conf.MODELNAME: None,
	Conf.IMAGE_CAPTION: False,
	Conf.IMAGE_DIM: 50


}
