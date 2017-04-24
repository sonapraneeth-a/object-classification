import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


class Summaries:

    def __init__(self,
                 train=True,
                 validate=False,
                 test=False,
                 verbose=False):
        self.train = train
        self.validate = validate
        self.test = test
        self.summary_writer = None
        self.verbose = verbose

    def writer(self, graph, log_dir='./tensorboard_logs/'):
        self.summary_writer = \
            tf.summary.FileWriter(log_dir, graph=graph)
        return True

    def write_loss(self, loss_value, list_loss, name=''):
        var_loss = tf.Variable([], dtype=tf.float32, trainable=False,
                               validate_shape=False, name=name + '_loss_list')
        update_loss = tf.assign(var_loss, list_loss, validate_shape=False)
        summary_loss = tf.summary.scalar(name+'_error', loss_value)
        return loss_value, var_loss, update_loss, summary_loss

    def write_accuracy(self, predictions, list_acc, name=''):
        acc = tf.reduce_mean(tf.cast(predictions, tf.float32))
        var_acc = tf.Variable([], dtype=tf.float32, trainable=False,
                                    validate_shape=False, name=name+'_accuracy_list')
        update_acc = tf.assign(var_acc, list_acc, validate_shape=False)
        summary_acc = tf.summary.scalar(name+'_accuracy', acc)
        return acc, var_acc, update_acc, summary_acc

    def write_learn_rate(self, learn_rate, list_learn_rate):
        var_learn_rate = tf.Variable([], dtype=tf.float32, trainable=False,
                                     validate_shape=False, name='learning_rate_progress')
        update_learn_rate = tf.assign(var_learn_rate, list_learn_rate, validate_shape=False)
        summary_learn_rate = tf.summary.scalar('learning_rate', learn_rate)
        return var_learn_rate, update_learn_rate, summary_learn_rate


class FreezeGraph:

    def __init__(self, verbose):
        self.verbose = verbose
        self.dir = os.path.dirname(os.path.realpath(__file__))

    def freeze_graph(self):
        return True

    def load_graph(self):
        return True