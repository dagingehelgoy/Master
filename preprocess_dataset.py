from os import listdir
from os.path import isfile, join, isdir

import numpy as np
# from keras import backend as K
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.models import Model
# from keras.preprocessing import image
# K.set_image_dim_ordering('tf')
# LAYER = 'flatten'

from data.database.helpers.caption_database_helper import save_caption_vector_list
from data.database.helpers.class_database_helper import save_class_vector_list
from data.database.helpers.image_database_helper import store_image_vector_to_db
from helpers.list_helpers import print_progress


def get_model():
	model = InceptionV3(weights='imagenet')
	model = Model(input=model.input, output=model.get_layer(LAYER).output)
	return model


# return Model(input=base_model.input, output=base_model.get_layer(LAYER).output)

counter = 0


def preprocess(img_path, num_images):
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	global counter
	print_progress(counter, num_images, prefix="Processing images")
	counter += 1
	return x


def fetch_all_filenames(dirpath):
	return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]


def fetch_class_dirs(flowers_text_path):
	all_files = listdir(flowers_text_path)
	return [f for f in all_files if isdir(join(flowers_text_path, f))]


def run_inception():
	flowers_path = "/Users/markus/workspace/master/Master/data/datasets/flowers/"
	flowers_image_path = flowers_path + "jpg/"

	image_filenames = fetch_all_filenames(flowers_image_path)
	num_images = len(image_filenames)
	print "Fetch image paths"
	image_paths = [flowers_image_path + image_filename for image_filename in image_filenames]
	processed_imgs = [preprocess(x, num_images) for x in image_paths]
	print "Loading model"
	inception = get_model()
	for i in range(num_images):
		img_vec = inception.predict(np.asarray(processed_imgs[i]))
		store_image_vector_to_db(image_filenames[i], img_vec[0])
		print_progress(i, num_images, prefix="Running images through model and storing...")


def save_classes():
	flowers_path = "/Users/markus/workspace/master/Master/data/datasets/flowers/"
	flowers_text_path = flowers_path + "text/"
	class_dirs = fetch_class_dirs(flowers_text_path)
	for class_dir in class_dirs:
		class_filesnames = fetch_all_filenames(flowers_text_path + class_dir)
		class_name = class_dir[6:]
		tuples = []
		for filename in class_filesnames:
			if filename != 'class.txt':
				filename = filename[:-4]
				tuples.append((filename, class_name, None))
		save_class_vector_list(tuples)


def save_captions():
	flowers_path = "/Users/markus/workspace/master/Master/data/datasets/flowers/"
	flowers_text_path = flowers_path + "text/"
	class_dirs = fetch_class_dirs(flowers_text_path)
	for class_dir in class_dirs:
		filesnames = fetch_all_filenames(flowers_text_path + class_dir)
		for filename in filesnames:
			if filename != 'class.txt':
				textfile = open(flowers_text_path + class_dir + "/" + filename, 'r')
				caption_lines = textfile.readlines()
				textfile.close()
				filename = filename[:-4]
				captions_tuples = []
				for line in caption_lines:
					captions_tuples.append((filename, line.strip().replace(",", "").replace(".", ""), None))
				save_caption_vector_list(captions_tuples)


if __name__ == "__main__":
	# run_inception()
	# save_classes()
	save_captions()
