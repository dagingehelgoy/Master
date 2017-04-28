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
	# Conf.NAME_SUFFIX: "cnn",
	# Conf.NAME_SUFFIX: "sam1",

	Conf.VOCAB_SIZE: 1000,
	Conf.MAX_SEQ_LENGTH: 8,

	# Conf.LIMITED_DATASET: "flowers",
	Conf.LIMITED_DATASET: "00058.txt",
	# Conf.LIMITED_DATASET: "two_flowers.txt",
	# Conf.LIMITED_DATASET: "person_surf.txt",
	# Conf.LIMITED_DATASET: None,
	Conf.DATASET_SIZE: -1,
	Conf.BATCH_SIZE: 64,
	Conf.EPOCHS: 10000000,

	Conf.NOISE_MODE: NoiseMode.REPEAT,

	Conf.MAX_LOSS_DIFF: 0,

	Conf.EMBEDDING_SIZE: 50,
	Conf.NOISE_SIZE: 50,
	Conf.PREINIT: PreInit.NONE,

	Conf.WORD2VEC_NUM_STEPS: 100001,

	Conf.MODELNAME: "2017-04-25_ImgCapFalse_word2vec_Vocab1000_Seq8_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_00058",
	# Conf.MODELNAME: None,

	Conf.IMAGE_CAPTION: False,
	Conf.IMAGE_DIM: 50

}
