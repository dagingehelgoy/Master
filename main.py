import argparse
import keras.backend.tensorflow_backend as ker

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--code", type=str)
	parser.add_argument("--inference", action='store_true')
	parser.add_argument("--env", type=str)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	config1 = ker.tf.ConfigProto()
	config1.gpu_options.allow_growth = True
	ker.set_session(ker.tf.Session(config=config1))
	args = get_args()
	if args.code == "seq2seq":
		from sequence_to_sequence.seq2seq import seq2seq
		seq2seq(args.inference)
	elif args.code == "one_hot_seq2seq":
		from sequence_to_sequence.one_hot_seq2seq import seq2seq
		seq2seq(args.inference)
	elif args.code == "genclass":
		from data.database.helpers.class_database_helper import gen_class_embs
		gen_class_embs()

	# elif args.code == "char_lstm":
	# 	from sequence_to_sequence.character_LSTM import char_lstm
	# 	char_lstm()
	# elif args.code == "mtm_word":
	# 	from sequence_to_sequence.mtm_word_lstm import word_lstm
	# 	word_lstm(args.inference)
	# elif args.code == "mts_word":
	# 	from sequence_to_sequence.mts_word_lstm import word_lstm
	# 	word_lstm(args.inference)

	else:
		print("### No suitable --code ###")

