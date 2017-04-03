#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
cachedStopWords = stopwords.words("english")


# Word Preprosessing

def preprocessing(text, remove_stopwords=False, stem_word=False, min_length=-1, add_sos_eos=False):
	# Tokenizing
	words = tokenize(text)

	# ascii_check(words)

	if remove_stopwords:
		words = stopwording(words)

	if stem_word:
		words = stemming(words)

	if min_length > 0:
		words = filtering(words, min_length)

	return words


def ascii_check(words):
	for word in words:
		try:
			word.decode('ascii')
		except UnicodeDecodeError:
			print "it was not a ascii-encoded unicode string"
		else:
			print "It may have been an ascii-encoded unicode string"


def tokenize(text):
	words = ["<sos>"]
	words.extend([x.lower() for x in text.split()])
	words.append("<eos>")
	words.append("<pad>")
	# words = map(lambda word: word.lower(), word_tokenize(text))
	return words


def stopwording(words):
	words = [word for word in words
	         if word not in cachedStopWords]
	return words


def stemming(words):
	words = (list(map(lambda token: PorterStemmer().stem(token),
	                  words)))
	return words


def filtering(words, min_length):
	p = re.compile('[a-zA-Z]+')
	# Filtering
	words = list(filter(lambda token:
	                    p.match(token) and len(token) >= min_length,
	                    words))
	return words


if __name__ == "__main__":
	print(preprocessing("Five"))
