import argparse

from GAN_caption.gan import gan_main


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--code", type=str)
	parser.add_argument("--env", type=str)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()
	if args.code == "gan":
		gan_main(args)
