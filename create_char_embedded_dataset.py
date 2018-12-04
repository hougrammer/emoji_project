from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.utils import plot_model
import numpy as np
import random
import sys
import csv
import os
import h5py
import time

embeddings_path = "/home/michellecutler/w266/char-embeddings-master/glove.840B.300d-char.txt"
embedding_dim = 300
batch_size = 128
use_pca = False
# lr = 0.001
# lr_decay = 1e-4
maxlen = 40
# consume_less = 2   # 0 for cpu, 2 for gpu

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
print('===================')
print('===================')
print('Processing pretrained character embeds...')
embedding_vectors = {}
with open(embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

print("Available characters for embedding")
print("number of characters in dictionary = ", len(embedding_vectors))
print("number of dimensions = ", len(embedding_vectors["H"]))
print('embedding_vectors.keys() = ', embedding_vectors.keys())

keys=list(embedding_vectors.keys())  #in python 3, you'll need `list(i.keys())`
values=embedding_vectors.values()


print()
print('===================')
print('===================')
print("Importing Barbieri dataset into list of list of characters")
# Import the Barbieri data from CSV into a list of lines
# entry = [sentence, emoji]
import csv

entries_train = []
with open('../AreEmojisPredictableData/5_train', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for entry in reader:
        entries_train.append(entry)
        
entries_test = []
with open('../AreEmojisPredictableData/5_test', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for entry in reader:
        entries_test.append(entry)
        
entries_validation = []
with open('../AreEmojisPredictableData/5_validation', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for entry in reader:
        entries_validation.append(entry)
        
print("size of entries_train = ", len(entries_train))
print("size of entries_test = ", len(entries_test))
print("size of entries_validation = ", len(entries_validation))


print()
print("Preparing data for character embedding")
# Remove the emoji from the entries
sentences_train = [entry[0] for entry in entries_train]
# Make all the characters one big blob - combine list of char strings into one giant char string
char_train = '_'.join(sentences_train)

sentences_test = [entry[0] for entry in entries_test]
# Make all the characters one big blob - combine list of char strings into one giant char string
char_test = '_'.join(sentences_test)

sentences_validation = [entry[0] for entry in entries_validation]
# Make all the characters one big blob - combine list of char strings into one giant char string
char_validation = '_'.join(sentences_validation)


emojis_train = [entry[1] for entry in entries_train]
emojis_test = [entry[1] for entry in entries_test]
emojis_validation = [entry[1] for entry in entries_validation]


print()
print("Converting dataset into character representation")
# Convert text into character representation
# Wants all characters from samples bundled together

# Reduce amount of data
text = char_train[:int(len(char_train)/3)]
print('corpus length:', len(text))


chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print()
print("Use indices_char[NUM] to lookup character by ")
print("indices_char[3] ", indices_char[3])
print("indices_char[6] ", indices_char[6])
print("indices_char ", indices_char)
print()

print("char_indices ", char_indices)
print()


MAX_SENT_LENGTH = 40
print("Trimming tweets to MAX_SENT_LENGTH = ", MAX_SENT_LENGTH)
sent_trim_train = [sentence[0:min(len(sentence),MAX_SENT_LENGTH)] for sentence in sentences_train]
sent_trim_test = [sentence[0:min(len(sentence),MAX_SENT_LENGTH)] for sentence in sentences_test]
sent_trim_validation = [sentence[0:min(len(sentence),MAX_SENT_LENGTH)] for sentence in sentences_validation]


print("sent_trim_train[0] ", sent_trim_train[0])
print("len(sent_trim_train[0]) ", len(sent_trim_train[0]))

print("sent_trim_test[0] ", sent_trim_test[0])
print("len(sent_trim_test[0]) ", len(sent_trim_test[0]))

print("sent_trim_validation[0] ", sent_trim_validation[0])
print("len(sent_trim_validation[0]) ", len(sent_trim_validation[0]))
print()




embedding_matrix = np.zeros((len(chars), 300))
#embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
for char, i in char_indices.items():
    #print ("{}, {}".format(char, i))
    embedding_vector = embedding_vectors.get(char)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


sentences = sent_trim_train
sentences_test = sent_trim_test
sentences_validation = sent_trim_validation

print()
print('===================')
print('===================')
print("Vectorize the dataset (with padding)")
# X = np.zeros((len(sentences), maxlen), dtype=np.int)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         try:
#             X[i, t] = char_indices[char]
#         except:
#             print("char = ", char)
#     y[i, char_indices[next_chars[i]]] = 1
    
# print("len(X[0]) = ", len(X[0]))
# print("character embedding X[0] = ", X[0])

X_train = np.zeros((len(sentences), maxlen), dtype=np.int)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_train[i, t] = char_indices[char]
        
X_test = np.zeros((len(sentences_test), maxlen), dtype=np.int)
for i, sentence in enumerate(sentences_test):
    for t, char in enumerate(sentence):
        X_test[i, t] = char_indices[char]
        
X_validation = np.zeros((len(sentences_validation), maxlen), dtype=np.int)
for i, sentence in enumerate(sentences_validation):
    for t, char in enumerate(sentence):
        X_validation[i, t] = char_indices[char]

print("len(X_train[0]) = ", len(X_train[0]))
print("character embedding X_train[0] = ", X_train[0])

print("len(X_test[0]) = ", len(X_test[0]))
print("character embedding X_test[0] = ", X_test[0])

print("len(X_validation[0]) = ", len(X_validation[0]))
print("character embedding X_validation[0] = ", X_validation[0])
      
print("DONE vectorizing sentences")
print('tweets stored in X_train')
print()

print()
print('===================')
print('===================')
print("Vectorizing emojis")

from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 20

def tokenize_data(texts):
    '''
    texts - array of strings to tokenize
    '''
    # print texts
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # print sequences
    word_index = tokenizer.word_index
    
    return sequences, word_index

emojis_embedded_train, emoji_dict = tokenize_data(emojis_train)

#Flatten list of lists to list
emojis_embedded_train = np.asarray([emoji for sublist in emojis_embedded_train for emoji in sublist])

# Use emoji_dict to convert emoji_x from u to integer of corresponding dictionary
emojis_embedded_test = np.asarray([emoji_dict.get(emoji) for emoji in emojis_test])
emojis_embedded_validation = np.asarray([emoji_dict.get(emoji) for emoji in emojis_validation])


print()
print('emoji_dict ', emoji_dict)

print()
print('tweet emojis stored in emojis_embedded')
print('emojis_embedded_train[0:10]', emojis_embedded_train[0:10])
print('emojis_embedded_test[0:10]', emojis_embedded_test[0:10])
print('emojis_embedded_validation[0:10]', emojis_embedded_validation[0:10])
print()



print()
print('===================')
print('===================')
print("One hot encoding all vectors X and y")

from numpy import array
from numpy import argmax
import numpy as np
from keras.utils import to_categorical


# Can't take more than 58,689 samples right now...
# NUM_SAMPLES = int(len(X_train)/5)
# NUM_SAMPLES = 100000
NUM_SAMPLES = len(X_train)


print('selecting from dataset NUM_SAMPLES = ', NUM_SAMPLES)

def one_hot_encode(_X, _sequence_length, _NUM_SAMPLES, _vocab_size):
    _X_encode = np.zeros((_NUM_SAMPLES, _sequence_length, _vocab_size))
    for i, data in enumerate(_X[:_NUM_SAMPLES]):
        _X_encode[i,:,:] = to_categorical(data-1,
                                num_classes=_vocab_size)
    return _X_encode

    
X_enc_train = one_hot_encode(X_train, 40, NUM_SAMPLES, len(indices_char))
# print(X_enc_train.shape)

X_enc_test = one_hot_encode(X_test, 40, len(X_test), len(indices_char))
# print(X_enc_test.shape)

X_enc_validation  = one_hot_encode(X_validation, 40, len(X_validation), len(indices_char))
# print(X_enc_validation.shape)

num_emojis = 5

y_enc_train = one_hot_encode(emojis_embedded_train, 1, NUM_SAMPLES, num_emojis)
# print(y_enc_train.shape)

y_enc_test = one_hot_encode(emojis_embedded_test, 1, len(emojis_embedded_test), num_emojis)
# print(y_enc_train.shape)

y_enc_validation = one_hot_encode(emojis_embedded_validation, 1, len(emojis_embedded_validation), num_emojis)
# print(y_enc_train.shape)

print("Use X_enc_train, X_enc_test, X_enc_validation, y_enc_train, y_enc_test, y_enc_validation")
print("DONE")