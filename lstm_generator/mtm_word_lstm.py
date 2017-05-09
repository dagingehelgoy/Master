import itertools
import random
import string

import keras
import numpy
import sys
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Embedding, TimeDistributed, RepeatVector, Lambda
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from helpers.io_helper import load_pickle_file, check_pickle_file, save_pickle_file
from helpers.list_helpers import print_progress


def open_corpus():
    sentence_file = open("data/datasets/all_flowers.txt")
    lines = sentence_file.readlines()
    sentence_file.close()
    return lines


def batch_generator(data, batch_size, n_vocab):
    while 1:
        for pos in range(0, len(data), batch_size):
            Xs = []
            Ys = []
            for i in range(pos, pos + batch_size):
                X = data[i]

                Xs.append(X)

                Y_sentence = data[pos]

                Y_onehot = []

                for y_index in range(1, len(Y_sentence)):
                    onehot = numpy.zeros(n_vocab)
                    onehot[Y_sentence[y_index]] = 1
                    Y_onehot.append(onehot)

                zero_onehot = numpy.zeros(n_vocab)
                zero_onehot[0] = 1
                Y_onehot.append(zero_onehot)
                Ys.append(Y_onehot)
            yield (numpy.asarray(Xs), numpy.asarray(Ys))


def get_word_embedding_matrix(word_to_id, embedding_dim):
    embeddings_dict = load_pickle_file('word2vec/saved_models/word2vec_50d1000voc100001steps_dict_flowers.pkl')
    embedding_matrix = numpy.zeros((len(word_to_id) + 1, embedding_dim))
    for word, i in word_to_id.items():
        if word in embeddings_dict:
            embedding_matrix[i] = embeddings_dict[word]


def get_model(nb_words, embedding_layer):
    # define the LSTM model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(20))
    model.add(Dense(nb_words, activation='softmax'))
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='adam')
    return model


def word_lstm(train=True):
    NB_WORDS = 5000
    EPOCHS = 200
    BATCH_SIZE = 32
    VAL_DATA_SIZE = 3000
    MAX_SEQUENCE_LENGTH = 20
    EMBEDDING_DIMENSION = 50
    FLICKR = False

    log_folder = "mtm_1-512-lstm_2-2d_32-bs_pre-pad_rsFalse"
    log_dir = "text_generators/logs/"
    if not os.path.exists(log_dir + log_folder):
        os.makedirs(log_dir + log_folder)

    lines = open_corpus()

    ######### Tokenizer #########
    if FLICKR:
        lines = [(line.split("\t")[1]).strip() for line in lines]
    print
    lines = ['<sos> ' + line + ' <eos>' for line in lines]
    tokenizer = Tokenizer(nb_words=NB_WORDS, filters="""!"#$%&'()*+-/:;=?@[\]^_`{|}~""")
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    word_to_id = tokenizer.word_index
    id_to_word = {token: idx for idx, token in word_to_id.items()}



    print("Sequence length: %s" % MAX_SEQUENCE_LENGTH)
    print("Nb words: %s" % NB_WORDS)

    ######### Embeddings #########
    # 2_lstm_many_to_many_post_pad_glove
    # Train:
    embedding_layer = Embedding(NB_WORDS + 1, EMBEDDING_DIMENSION, trainable=False, input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, weights=get_word_embedding_matrix(word_to_id, EMBEDDING_DIMENSION))
    #embedding_layer = Embedding(NB_WORDS + 1, EMBEDDING_DIMENSION, input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, weights=get_word_embedding_matrix(word_to_id, EMBEDDING_DIMENSION))

    # 2_lstm_many_to_many_post_pad
    # embedding_layer = Embedding(NB_WORDS + 1, EMBEDDING_DIMENSION, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=True)

    model = get_model(NB_WORDS, embedding_layer)

    if train:
        # define the checkpoint
        filepath = "text_generators/weights/" + log_folder + "--weights-{loss:.4f}.hdf5"
        data = sequence.pad_sequences(sequences, MAX_SEQUENCE_LENGTH)
        numpy.random.shuffle(data)
        val_gen = batch_generator(data[-VAL_DATA_SIZE:], BATCH_SIZE, NB_WORDS)
        train_gen = batch_generator(data[:-VAL_DATA_SIZE], BATCH_SIZE, NB_WORDS)

        # Check if model is running on server
        if 'SSH_CONNECTION' in os.environ.keys():
            tensorboard = keras.callbacks.TensorBoard(log_dir='text_generators/logs/' + log_folder, histogram_freq=1, write_graph=True)
            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=3)
            callbacks_list = [checkpoint, tensorboard, es]
            model.fit_generator(generator=train_gen, samples_per_epoch=len(data)-VAL_DATA_SIZE, validation_data=val_gen, nb_val_samples=VAL_DATA_SIZE, nb_epoch=EPOCHS, callbacks=callbacks_list)
        else:
            from keras.utils.visualize_util import plot
            plot(model, show_shapes=True, to_file=log_dir + log_folder + "/model.png")
            model.fit_generator(train_gen, len(data), EPOCHS)


    else:
        loss = "2.8836"
        # filename = "text_generators/weights/" + log_folder + "--weights-" + loss + ".hdf5"
        filename = "text_generators/weights/2_lstm_many_to_many_post_pad--weights-0.4096.hdf5"
        model.load_weights(filename)

        # Random seed = 0, Single input = 1, Loop predict = 2
        testing_mode = 0

        if testing_mode == 0:
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print
                print '----- diversity: %s' % diversity
                generated = ''
                sentence = sequences[random.randint(0, len(sequences))][:MAX_SEQUENCE_LENGTH]
                sentence_string = ""
                for word_id in sentence:
                    sentence_string += id_to_word[word_id] + '(' + str(word_id) + ') '
                generated += sentence_string
                print '----- Generating with seed: %s' % sentence_string
                print

                for i in range(20):
                    x = numpy.zeros(MAX_SEQUENCE_LENGTH, dtype='int64')
                    for t, word_id in enumerate(sentence):
                        x[t] = word_id
                    x = numpy.asarray([x])
                    preds = model.predict(x, verbose=0)[0][0]
                    next_index = sample(preds, diversity)
                    if next_index == 0:
                        next_word = 'null'
                    else:
                        next_word = id_to_word[next_index]
                    generated += next_word
                    del sentence[0]
                    sentence.append(next_index)
                    sys.stdout.write(' ')
                    print_string = next_word + " (" + str(next_index) + ")"
                    sys.stdout.write(print_string)
                    sys.stdout.flush()
                print
        else:
            while True:
                start_sentence = raw_input("\nStart sentence... ")
                start_sentence = [x.lower() for x in start_sentence.split(" ")]

                start_sentence = start_sentence[:20]

                start_sentence = [word_to_id[word] for word in start_sentence]

                for id in start_sentence:
                    sys.stdout.write(id_to_word[id] + " ")

                while len(start_sentence) < MAX_SEQUENCE_LENGTH:
                    start_sentence = start_sentence + [0]

                start_sentence = [start_sentence]

                if testing_mode == 1:
                    preds = model.predict(numpy.asarray(start_sentence), verbose=0)[0]
                    for pred in preds:
                        argmax = numpy.argmax(pred)
                        if argmax == 0:
                            sys.stdout.write("0" + " ")
                        else:
                            sys.stdout.write(id_to_word[argmax] + " ")

                elif testing_mode == 2:
                    for i in range(20):
                        # prepared_sentence = prepare_data(start_sentence, tokenizer.nb_words, MAX_SEQUENCE_LENGTH)

                        preds = model.predict(numpy.asarray(start_sentence), verbose=0)[0]

                        argmax = numpy.argmax(preds)
                        # argmax_normalized = float(argmax) / len(word_to_id)
                        start_sentence = [numpy.append(start_sentence[0][1:], argmax)]
                        if argmax == 0:
                            sys.stdout.write("0" + " ")
                        else:
                            sys.stdout.write(id_to_word[argmax] + " ")
                        sys.stdout.flush()
                        if argmax == 356:
                            break


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = numpy.log(a) / temperature
    a = numpy.exp(a) / numpy.sum(numpy.exp(a))
    a *= 0.5
    return numpy.argmax(numpy.random.multinomial(1, a, 1))
