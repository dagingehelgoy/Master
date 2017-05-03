import random

from PIL import Image, ImageDraw, ImageFont

from data.database.helpers.caption_database_helper import fetch_caption_texts_for_image_name
from data.database.helpers.image_database_helper import *
from data.database.helpers.pca_database_helper import fetch_all_pca_vector_pairs, fetch_pca_vector
from helpers.list_helpers import *


def show_image(file, title, index, mode):
	try:
		title = "%s_%s_%s" % (mode, index, title)

		img = Image.open(file)
		# draw = ImageDraw.Draw(img)
		# font = ImageFont.truetype(settings.RES_DIR + "Montserrat-Bold.ttf", 10)
		# draw.text((0, 0), title, (0, 0, 0), font=font)
		# draw.text((1, 1), title, (255, 255, 255), font=font)
		# img.show(title=title)
		img.save("pics/%s" % title)
	except Exception as e:
		print("filename:", file)
		print(e)


def test_image_vectors():
	test_size = 1
	num_similar_images = 5
	all_image_names = fetch_all_image_names()
	# all_image_names = ["3331900249_5872e90b25.jpg"]
	np.random.shuffle(all_image_names)
	start = random.randint(0, len(all_image_names) - test_size)
	samples = all_image_names[start:start + test_size]
	image_vector_pairs = fetch_all_image_vector_pairs()

	print "All images %s" % len(image_vector_pairs)
	print("\nRESULTS")
	for i in range(len(samples)):
		correct_image_name = all_image_names[i:i + 1][0]

		correct_image_vector = fetch_image_vector(correct_image_name)

		first_image_filename = image_vector_pairs[0][0]
		first_image_vector = image_vector_pairs[0][1]
		first_image_mse = compare_vectors(correct_image_vector, first_image_vector)

		best_image_vector_mse_list = [0 for _ in range(num_similar_images)]
		best_image_vector_name_list = ["" for _ in range(num_similar_images)]
		best_image_vector_list = [[] for _ in range(num_similar_images)]

		best_image_vector_mse_list = insert_and_remove_last(0, best_image_vector_mse_list, first_image_mse)
		best_image_vector_name_list = insert_and_remove_last(0, best_image_vector_name_list, first_image_filename)
		best_image_vector_list = insert_and_remove_last(0, best_image_vector_list, first_image_vector)

		for temp_image_name, temp_image_vector in image_vector_pairs:
			temp_image_vector_mse = compare_vectors(correct_image_vector, temp_image_vector)
			for index in range(len(best_image_vector_list)):
				if temp_image_vector_mse < best_image_vector_mse_list[index]:
					best_image_vector_mse_list = insert_and_remove_last(index, best_image_vector_mse_list,
					                                                    temp_image_vector_mse)
					best_image_vector_name_list = insert_and_remove_last(index, best_image_vector_name_list,
					                                                     temp_image_name)
					best_image_vector_list = insert_and_remove_last(index, best_image_vector_list,
					                                                temp_image_vector)
					break
		print("")
		print("Correct filename:\t", correct_image_name)
		print("")
		print("Most similar images(chosen using image vectors):")
		for j in range(len(best_image_vector_mse_list)):
			filename = best_image_vector_name_list[j]
			show_image(settings.IMAGE_DIR + filename, filename, j, "inc")
			print(j + 1, filename)
		print("")
		for name in best_image_vector_name_list:
			caps = fetch_caption_texts_for_image_name(name)
			for cap in caps:
				print cap
				# print "<sos> %s" % " ".join(cap.split(" ")[:9])


def test_pca_vectors():
	test_size = 1
	all_image_names = fetch_all_image_names()
	np.random.shuffle(all_image_names)
	start = random.randint(0, len(all_image_names) - test_size)
	samples = all_image_names[start:start + test_size]
	pca_vector_pairs = fetch_all_pca_vector_pairs()
	print "All images %s" % len(pca_vector_pairs)
	print("\nRESULTS")
	for i in range(len(samples)):
		correct_image_name = all_image_names[i:i + 1][0]

		correct_pca_vector = fetch_pca_vector(correct_image_name)
		print len(correct_pca_vector)

		first_image_filename = pca_vector_pairs[0][0]
		first_image_vector = pca_vector_pairs[0][1]
		first_image_mse = compare_vectors(correct_pca_vector, first_image_vector)

		best_image_vector_mse_list = [0 for _ in range(5)]
		best_image_vector_name_list = ["" for _ in range(5)]
		best_image_vector_list = [[] for _ in range(5)]

		best_image_vector_mse_list = insert_and_remove_last(0, best_image_vector_mse_list, first_image_mse)
		best_image_vector_name_list = insert_and_remove_last(0, best_image_vector_name_list, first_image_filename)
		best_image_vector_list = insert_and_remove_last(0, best_image_vector_list, first_image_vector)

		for temp_image_name, temp_image_vector in pca_vector_pairs:
			temp_image_vector_mse = compare_vectors(correct_pca_vector, temp_image_vector)
			for index in range(len(best_image_vector_list)):
				if temp_image_vector_mse < best_image_vector_mse_list[index]:
					best_image_vector_mse_list = insert_and_remove_last(index, best_image_vector_mse_list,
					                                                    temp_image_vector_mse)
					best_image_vector_name_list = insert_and_remove_last(index, best_image_vector_name_list,
					                                                     temp_image_name)
					best_image_vector_list = insert_and_remove_last(index, best_image_vector_list,
					                                                temp_image_vector)
					break
		print("")
		print("Correct filename:\t", correct_image_name)
		print("")
		print("Most similar images(chosen using image vectors):")
		for j in range(len(best_image_vector_mse_list)):
			filename = best_image_vector_name_list[j]
			show_image(settings.IMAGE_DIR + filename, filename, j, "pca")
			print(j + 1, filename)
		print("")


if __name__ == '__main__':
	# test_image_vectors()
	test_pca_vectors()
