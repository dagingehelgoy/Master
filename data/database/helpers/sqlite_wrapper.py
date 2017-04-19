import io
import sqlite3

import numpy as np

import settings

def adapt_array(arr):
	""" http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())


def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)


db = sqlite3.connect(settings.DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
print "connecting to %s" % settings.DB_FILE_PATH
# TODO str Not working in python 2, unicode does
# db.text_factory = lambda x: unicode(x, "utf-8", "ignore")

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

c = db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images (filename TEXT UNIQUE, image_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS pca (filename TEXT UNIQUE, pca_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS captions (filename TEXT, caption_text TEXT, caption_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS words (word_text TEXT UNIQUE, word_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS classes (filename TEXT, class_text TEXT, class_vector array)''')
db.commit()


def update_database_connection(word_embedding, image_embedding):
	global db
	settings.DB_SUFFIX = "%s-%s-%s" % (image_embedding, word_embedding, settings.DATASET)
	settings.DB_FILE_PATH = settings.ROOT_DIR + "/data/databases/sqlite/data-%s.db" % settings.DB_SUFFIX
	db = sqlite3.connect(settings.DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
	settings.STORED_EMBEDDINGS_NAME = "%s-%s" % (settings.DB_SUFFIX, settings.NEG_TAG)
	settings.IMAGE_EMBEDDING_DIMENSIONS = 4096 if image_embedding == "vgg" else 2048
	print("Connected to %s" % settings.DB_FILE_PATH)


""" TABLE: WORDS """


def db_insert_word_vector(word_text, word_vector):
	cursor = db.cursor()
	cursor.execute("""INSERT INTO words VALUES(?, ?)""", (word_text, word_vector))
	db.commit()


def db_insert_word_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""INSERT INTO words VALUES (?, ?)""", tuple_list)
	db.commit()


def db_fetch_all_word_vectors():
	cursor = db.cursor()
	return cursor.execute("""SELECT word_text, word_vector FROM words""").fetchall()


def db_fetch_word_vector(word, default=None):
	cursor = db.cursor()
	result = cursor.execute("""SELECT word_vector FROM words WHERE word_text = ?""", (word,)).fetchone()
	if result is None:
		return default
	return result


""" TABLE: IMAGES """


def db_keys_images():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get_image_vector(filename, default=None):
	cursor = db.cursor()
	result = cursor.execute("""SELECT image_vector FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_all_filename_img_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, image_vector FROM images""").fetchall()


def db_update_filename_img_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, image_vector FROM images""").fetchall()


def db_insert_image_vector(filename, image_vector):
	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, image_vector))
	db.commit()


def db_get_filename_from_image_vector(image_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM images WHERE image_vector = ?""",
	                        (image_vector,)).fetchone()
	return result


def db_insert_image_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""UPDATE images SET image_vector = ? WHERE filename = ?""", tuple_list)
	db.commit()


""" TABLE: PCA VECTORS """


def db_get_pca_vector(filename, default=None):
	cursor = db.cursor()
	result = cursor.execute("""SELECT pca_vector FROM pca WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_all_filename_pca_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, pca_vector FROM pca""").fetchall()


def db_update_filename_pca_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, pca_vector FROM pca""").fetchall()


def db_insert_pca_vector(filename, image_vector):
	cursor = db.cursor()
	cursor.execute("""INSERT INTO pca VALUES (?,?)""", (filename, image_vector))
	db.commit()


def db_get_filename_from_pca_vector(image_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM pca WHERE pca_vector = ?""",
	                        (image_vector,)).fetchone()
	return result


def db_insert_pca_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""UPDATE pca SET pca_vector = ? WHERE filename = ?""", tuple_list)
	db.commit()


""" TABLE: CAPTIONS """


def db_keys_captions():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM captions""").fetchall()


def db_all_caption_text_tuples():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_text FROM captions""").fetchall()


def db_all_filename_caption_vector_tuple():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_vector FROM captions""").fetchall()


def db_all_caption_rows():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_vector, caption_text FROM captions""").fetchall()


def db_get_caption_vectors(filename):
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_get_caption_texts(filename):
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_text FROM captions WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_fetch_all_caption_vectors():
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions""").fetchall()
	return result


def db_insert_caption_vector(filename, caption_text, caption_vector):
	try:
		cursor = db.cursor()
		cursor.execute("""INSERT INTO captions VALUES (?,?,?)""", (filename, caption_text, caption_vector))
		db.commit()
	except sqlite3.ProgrammingError as e:
		print(filename, caption_text)
		print(e)


def db_insert_caption_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""INSERT INTO captions VALUES (?,?,?)""", tuple_list)
	db.commit()


def db_get_caption_text(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_filenames_from_caption_vector(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchall()
	return result


def db_get_filename_caption_tuple_from_caption_vector(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename,caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_caption_table_size():
	cursor = db.cursor()
	result = cursor.execute("""SELECT COUNT(*) FROM captions""").fetchone()[0]
	return result


""" TABLE: CLASSES """


def db_keys_classes():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM classes""").fetchall()


def db_all_filename_class_vector_tuple():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, class_vector FROM classes""").fetchall()


def db_all_class_rows():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, class_vector, class_text FROM classes""").fetchall()


def db_get_class_vectors(filename):
	cursor = db.cursor()
	result = cursor.execute("""SELECT class_vector FROM classes WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_get_class_texts(filename):
	cursor = db.cursor()
	result = cursor.execute("""SELECT class_text FROM classes WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_fetch_all_class_vectors():
	cursor = db.cursor()
	result = cursor.execute("""SELECT class_vector FROM classes""").fetchall()
	return result


def db_insert_class_vector(filename, class_text, class_vector):
	try:
		cursor = db.cursor()
		cursor.execute("""INSERT INTO classes VALUES (?,?,?)""", (filename, class_text, class_vector))
		db.commit()
	except sqlite3.ProgrammingError as e:
		print(filename, class_text)
		print(e)


def db_insert_class_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""INSERT INTO classes VALUES (?,?,?)""", tuple_list)
	db.commit()


def db_get_class_text(class_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT class_text FROM classes WHERE class_vector = ?""",
	                        (class_vector,)).fetchone()
	return result


def db_get_filenames_from_class_vector(class_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM classes WHERE class_vector = ?""",
	                        (class_vector,)).fetchall()
	return result


def db_get_filename_class_tuple_from_class_vector(class_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename,class_text FROM classes WHERE class_vector = ?""",
	                        (class_vector,)).fetchone()
	return result


def db_get_class_table_size():
	cursor = db.cursor()
	result = cursor.execute("""SELECT COUNT(*) FROM classes""").fetchone()[0]
	return result
