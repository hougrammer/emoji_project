"""LSTM model for tweet corpus

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

Uses a fused block cell as recommended in https://www.tensorflow.org/guide/performance/overview#rnn_performance

tf.contrib.rnn.LSTMBlockFusedCell 

To run:

$ python emoji_lm.py --data_path=./

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import tweet_reader
import util
import gensim

from tensorflow.python.client import device_lib

w2v = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "default","A type of model. Possible options are: default, test")
flags.DEFINE_string("data_path", "./", "Where the training/test data is stored.")
flags.DEFINE_string("save_path",'./simple-examples/models/',"Model output directory.")
flags.DEFINE_bool("use_fp16", True,"Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class DefConfig(object):
    """default config."""
    init_scale = 0.1 #This goes in the random uniform initializer
    learning_rate = 1.0
    max_grad_norm = 5 #the maximum permissible norm of the gradient
    num_layers = 2 
    num_steps = 20
    hidden_size = 200
    embedding_size = 300
    max_epoch = 4
    max_max_epoch = 13
    lr_decay = 0.5 #the decay of the learning rate for each epoch after "max_epoch"
    batch_size = 20 
    vocab_size = None #it can be defined but the default is the length of the training vocabulary

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1 
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    embedding_size = 300
    max_epoch = 1
    max_max_epoch = 1
    lr_decay = 0.5
    batch_size = 20
    vocab_size = None

class Input(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = tweet_reader.tw_producer(data, batch_size, num_steps, name=name)
        
class Model(object):
    """The model."""

    def __init__(self, is_training, config, input_, word_to_id):
        self._is_training = is_training
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.embedding_size
        vocab_size = config.vocab_size #This is defined before in Main() if no input is provided.
        self.word_to_id = word_to_id
    
        with tf.device("/cpu:0"):
            #based on https://stackoverflow.com/questions/45113130/how-to-add-new-embeddings-for-unknown-words-in-tensorflow-training-pre-set-fo
            pretrained_vocab = list(w2v.vocab.keys())
            pretrained_embs = w2v.vectors
            train_vocab = list(self.word_to_id.keys())
            only_in_train = list(set(train_vocab) - set(pretrained_vocab))
            vocab = only_in_train + pretrained_vocab #First only in train so it keeps the same order that it had originally

            # Set up tensorflow look up from string word to unique integer
            vocab_lookup = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(vocab),default_value=len(vocab))
            string_tensor = vocab_lookup.lookup(tf.map_fn(self.id_to_word,input_.input_data))

            # define the word embedding
            pretrained_embs = tf.get_variable(
              name="embs_pretrained",
              initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
              shape=pretrained_embs.shape,
              trainable=False)
            train_embeddings = tf.get_variable(
              name="embs_only_in_train",
              shape=[len(only_in_train), emb_size],
              initializer=tf.random_uniform_initializer(-0.04, 0.04))

            embedding = tf.concat([pretrained_embs, train_embeddings], axis=0)

            inputs = tf.nn.embedding_lookup(embeddings, string_tensor)

            #lo que necesito embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type()) 
                            #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        fused_rnn_cell = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)

        output, state = fused_rnn_cell(inputs, dtype=tf.float32)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        #Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False) #Learning rate
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),config.max_grad_norm)
        optimizer  = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
    
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)
    
    def id_to_word(self,id):
        tf.print(id)
        return [x[0] for x in self.word_to_id.items() if x[1] == id][0]

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name
                 
def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "default":
        config = DefConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    return config                 

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
                 
    train_data, valid_data, test_data, config.vocab_size, word_to_id_train = tweet_reader.tweets_raw_data(FLAGS.data_path)
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    
    with tf.name_scope("Train"):
        train_input = Input(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = Model(is_training=True, config=config, input_=train_input, word_to_id=word_to_id_train)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = Input(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = Model(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = Input(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = Model(is_training=False, config=eval_config,input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
        model.export_ops(name)
        metagraph = tf.train.export_meta_graph()

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
    for model in models.values():
        model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op,verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
