import argparse
import keras.backend.tensorflow_backend as ker

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--code", type=str)
	parser.add_argument("--inference", action='store_false')
	parser.add_argument("--env", type=str)
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	config1 = ker.tf.ConfigProto()
	config1.gpu_options.allow_growth = True
	ker.set_session(ker.tf.Session(config=config1))
	args = get_args()
	if args.code == "gan":
		from GAN_caption.gan import gan_main
		gan_main(args)
	elif args.code == "word_lstm_gan":
		from GAN_caption.word_lstm_gan import word_lstm_gan
		word_lstm_gan()
	elif args.code == "char_lstm":
		from text_generators.character_LSTM import char_lstm
		char_lstm()
	elif args.code == "mtm_word":
		from text_generators.mtm_word_lstm import word_lstm
		word_lstm(args.inference)
	elif args.code == "mts_word":
		from text_generators.mts_word_lstm import word_lstm
		word_lstm(args.inference)
	elif args.code == "seq2seq":
		from text_generators.seq2seq import seq2seq
		seq2seq(args.inference)
	elif args.code == "one_hot_seq2seq":
		from text_generators.one_hot_seq2seq import seq2seq
		seq2seq(args.inference)
	else:
		print("### No suitable --code ###")

