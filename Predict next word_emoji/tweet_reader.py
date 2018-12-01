from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import re
import tensorflow as tf

regex = r"\\[U]\d{4}[f].{3}"
regex1 = r"[/\.!?]"
regex2 = r"[u].{4}"
subst = " <emoji> "

def rnnlm_batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form for RNN language modeling."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) // batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in range(0, input_w.shape[1], max_time):
        yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return re.sub(regex2,' ',re.sub(regex1,' ',re.sub(r"[0-9]",' <number> ',re.sub(regex,subst,f.read().replace('\r\n',' <eos> '))))).split()
    
def _build_vocab(filename):
    data = _read_words(filename)
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word for word in data if word in word_to_id]

def tweets_raw_data(data_path = None):
    train_path = os.path.join(data_path, "train.csv")
    valid_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_length = len(word_to_id)
    return train_data, valid_data, test_data, word_to_id

def tw_producer(raw_data, batch_size, num_steps, name=None):
    '''  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.'''

    with tf.name_scope(name, "tw_producer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.string)
        
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],[batch_size, batch_len]) #Reexpresses data len as the product of batch len and batch_size
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],[batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],[batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        
        return x, y
