import argparse
import keras.backend.tensorflow_backend as ker



def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--code", type=str)
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
	if args.code == "genclass":
		from data.database.helpers.class_database_helper import gen_class_embs
		gen_class_embs()
