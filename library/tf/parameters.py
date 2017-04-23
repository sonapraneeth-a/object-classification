import tensorflow as tf


class List:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set(self):
        return True

    def get(self):
        return True


class Weights:
    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def define(shape,
               weight_type='random_normal',
               mean=0.0, std=0.01,
               weight_name='Weight',
               scope=None):
        if scope is None:
            if weight_type == 'zeros':
                weight = tf.Variable(tf.zeros([shape[0], shape[1]]), name=weight_name)
            elif weight_type == 'ones':
                weight = tf.Variable(tf.ones([shape[0], shape[1]]), name=weight_name)
            elif weight_type == 'random_normal':
                weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
            else:
                weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
        else:
            with tf.name_scope(scope):
                if weight_type == 'zeros':
                    weight = tf.Variable(tf.zeros([shape[0], shape[1]]), name=weight_name)
                elif weight_type == 'ones':
                    weight = tf.Variable(tf.ones([shape[0], shape[1]]), name=weight_name)
                elif weight_type == 'random_normal':
                    weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
                else:
                    weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
        return weight


class Bias:
    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def define(shape, bias_type='random_normal', bias_name='Bias'):
        if bias_type == 'zeros':
            bias = tf.Variable(tf.zeros([shape[0]]), name=bias_name)
        elif bias_type == 'ones':
            bias = tf.Variable(tf.ones([shape[0]]), name=bias_name)
        elif bias_type == 'random_normal':
            bias = tf.Variable(tf.random_normal([shape[0]]), name=bias_name)
        else:
            bias = tf.Variable(tf.random_normal([shape[0]]), name=bias_name)
        return bias


class Variable:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set(self):
        return True

    def get(self):
        return True