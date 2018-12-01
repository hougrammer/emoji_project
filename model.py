import numpy as np
import pickle
import utils
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def with_self_graph(function):
    '''
    Graph wrapper
    '''
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class EmojiLSTM(object):
    def __init__(self, params=None):
        if params is None: params = {}
        
        self.graph = params.get('graph', tf.Graph())
        self.batch_size = params.get('batch_size', 128)
        self.hidden_size = params.get('hidden_size', 300)
        self.embed_size = params.get('embed_size', 300)
        self.x_seq_length = params.get('x_seq_length', 32)
        self.learning_rate = params.get('learning_rate', .001)
        self.use_dropout = params.get('use_dropout', False)
        
    
    @with_self_graph
    def build_graph(self):
        inputs = tf.placeholder(tf.float32, (None, x_seq_length, embed_size), 'inputs')
        input_mean = tf.nn.l2_normalize(tf.reduce_mean(inputs, axis=1), axis=1, name='input_mean')

        output_embedding = tf.constant(emoji_embedding, name='output_embedding')

        with tf.name_scope('network'):
            lstm_cell = tf.contrib.rnn.LSTMCell(nodes, name='lstm')
            lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs, dtype=tf.float32)

            logits = tf.layers.dense(lstm_outputs, units=len(emoji_vectors), activation='softmax', name='dense') 
            outputs = utils.matmul3d(logits, output_embedding)

            output_mean = tf.nn.l2_normalize(tf.reduce_mean(outputs, axis=1), axis=1)

        with tf.name_scope("optimization"):
            loss = tf.losses.cosine_distance(input_mean, output_mean, axis=1)
            optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('models/lstm_moby_dick', sess.graph)