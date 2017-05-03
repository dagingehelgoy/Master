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

	# Conf.NAME_SUFFIX: None,
	Conf.NAME_SUFFIX: "neg_examples",
	# Conf.NAME_SUFFIX: "sam1",

	Conf.VOCAB_SIZE: 1000,
	Conf.MAX_SEQ_LENGTH: 15,

	# Conf.LIMITED_DATASET: "flowers",
	# Conf.LIMITED_DATASET: "00058.txt",
	# Conf.LIMITED_DATASET: "two_flowers.txt",
	# Conf.LIMITED_DATASET: "person_surf.txt",
	# Conf.LIMITED_DATASET: "all_flowers.txt",
	Conf.LIMITED_DATASET: None,
	Conf.DATASET_SIZE: -1,
	Conf.BATCH_SIZE: 64,
	Conf.EPOCHS: 10000000,

	Conf.NOISE_MODE: NoiseMode.REPEAT,

	Conf.MAX_LOSS_DIFF: 0,

	Conf.EMBEDDING_SIZE: 50,
	Conf.NOISE_SIZE: 50,
	Conf.PREINIT: PreInit.NONE,

	Conf.WORD2VEC_NUM_STEPS: 100001,

	# Conf.MODELNAME: "2017-04-26_ImgCapFalse_WordEmbedding.WORD2VEC_Vocab1000_Seq10_Batch128_EmbSize50_NoiseMode.REPEAT_Noise50_PreInitPreInit.NONE_Dataset_all_flowers_500hidden_dropout0.2",
	Conf.MODELNAME: None,

	Conf.IMAGE_CAPTION: True,
	Conf.IMAGE_DIM: 50

}
