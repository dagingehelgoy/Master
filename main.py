import argparse

import keras.backend.tensorflow_backend as k_tf

from data.data_main import fetch_flower_captions


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--code", type=str)
	parser.add_argument("--model", type=str)
	parser.add_argument("--weights", type=str)
	parser.add_argument("--inference", action='store_true')
	parser.add_argument("--encode_data", action='store_true')
	parser.add_argument("--decode_random", action='store_true')
	parser.add_argument("--env", type=str)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	tf_config = k_tf.tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	k_tf.set_session(k_tf.tf.Session(config=tf_config))
	args = get_args()
	if args.code == "seq2seq":
		from sequence_to_sequence.embedding_seq2seq import seq2seq
		seq2seq(args.inference, args.encode_data, args.decode_random, args.model, args.weights)
	elif args.code == "one_hot_seq2seq":
		from sequence_to_sequence.one_hot_seq2seq import seq2seq
		seq2seq(args.inference, args.encode_data)
	elif args.code == "gan":
		from GAN.main import gan_main
		gan_main(args.inference)
	elif args.code == "compare_distributions":
		from word2vec.distribution_comparison import compare_distributions
		compare_distributions()
	elif args.code == "data":
		fetch_flower_captions()
	else:
		print("### No suitable --code ###")

