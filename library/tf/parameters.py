import tensorflow as tf
import math


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
               mean=0.0, std=0.02, factor=1.0,
               weight_name='Weight', seed=None,
               scope=None):
        if scope is None:
            if weight_type == 'zeros':
                weight = tf.Variable(tf.zeros(shape), name=weight_name)
            elif weight_type == 'ones':
                weight = tf.Variable(tf.ones(shape), name=weight_name)
            elif weight_type == 'random_normal':
                weight = tf.Variable(tf.random_normal(shape, mean=mean, stddev=std, seed=seed),
                                     name=weight_name)
            elif weight_type == 'truncated_normal':
                weight = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=std, seed=seed),
                                     name=weight_name)
            elif weight_type == 'uniform_scaling':
                input_size = 1.0
                for dim in shape[:-1]:
                    input_size *= float(dim)
                max_val = math.sqrt(3 / input_size) * factor
                weight = tf.Variable(tf.random_ops.random_uniform(shape, -max_val, max_val,
                                     seed=seed, name=weight_name))
            else:
                weight = tf.Variable(tf.random_normal(shape, mean=mean, stddev=std, seed=seed),
                                     name=weight_name)
        else:
            with tf.name_scope(scope):
                if weight_type == 'zeros':
                    weight = tf.Variable(tf.zeros(shape), name=weight_name)
                elif weight_type == 'ones':
                    weight = tf.Variable(tf.ones(shape), name=weight_name)
                elif weight_type == 'random_normal':
                    weight = tf.Variable(tf.random_normal(shape, mean=mean, stddev=std, seed=seed),
                                         name=weight_name)
                elif weight_type == 'truncated_normal':
                    weight = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=std, seed=seed),
                                         name=weight_name)
                else:
                    weight = tf.Variable(tf.random_normal(shape, mean=mean, stddev=std, seed=seed),
                                         name=weight_name)
        return weight


class Bias:
    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def define(shape,
               bias_type='random_normal',
               mean=0.0, std=1.0, seed=1,
               bias_name='Bias'):
        if bias_type == 'zeros':
            bias = tf.Variable(tf.zeros(shape), name=bias_name)
        elif bias_type == 'ones':
            bias = tf.Variable(tf.ones(shape), name=bias_name)
        elif bias_type == 'random_normal':
            bias = tf.Variable(tf.random_normal(shape), name=bias_name)
        elif bias_type == 'truncated_normal':
            bias = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=std, seed=seed),
                               name=bias_name)
        else:
            bias = tf.Variable(tf.random_normal(shape), name=bias_name)
        return bias


class Variable:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set(self):
        return True

    def get(self):
        return True