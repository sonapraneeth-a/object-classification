import tensorflow as tf


tf.set_random_seed(1)


def weights(shape, weight_type='random_normal', weight_name='weight', scope=None):
    if weight_type == 'zeros':
        weight = tf.Variable(tf.zeros([shape[0], shape[1]]), name=weight_name)
    elif weight_type == 'ones':
        weight = tf.Variable(tf.ones([shape[0], shape[1]]), name=weight_name)
    elif weight_type == 'random_normal':
        weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
    else:
        weight = tf.Variable(tf.random_normal([shape[0], shape[1]]), name=weight_name)
    return weight


def biases(shape, bias_type='random_normal', bias_name='bias'):
    if bias_type == 'zeros':
        bias = tf.Variable(tf.zeros([shape[0]]), name=bias_name)
    elif bias_type == 'ones':
        bias = tf.Variable(tf.ones([shape[0]]), name=bias_name)
    elif bias_type == 'random_normal':
        bias = tf.Variable(tf.random_normal([shape[0]]), name=bias_name)
    else:
        bias = tf.Variable(tf.random_normal([shape[0]]), name=bias_name)
    return bias


def activation_layer(input, activation_type='relu', activation_name=''):
    if activation_name == '':
        activation_name = activation_type
    if activation_type == 'sigmoid':
        layer = tf.nn.sigmoid(input, name=activation_name)
    elif activation_type == 'softmax':
        layer = tf.nn.softmax(input, name=activation_name)
    elif activation_type == 'relu':
        layer = tf.nn.relu(input, name=activation_name)
    else:
        layer = tf.nn.relu(input, name=activation_name)
    return layer


def mlp_layer(prev_layer, weight, bias, activation_type='sigmoid', layer_name='mlp_layer'):
    var = tf.add(tf.matmul(prev_layer, weight), bias)
    layer = activation_layer(var, activation_name=layer_name)
    return layer


def optimize_algo(learning_rate=0.001, descent_method='gradient',
                  adam_beta1=0.9, adam_beta2=0.9):
    if descent_method == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif descent_method == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=adam_beta1, beta2=adam_beta2)
    elif descent_method == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif descent_method == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif descent_method == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return optimizer


def convolve(input, weight, bias, strides=1, activation_type='relu', conv_layer_name=''):
    x = tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return activation_layer(x, activation_type=activation_type, activation_name=conv_layer_name)


def maxpool_layer(input, overlap=2, stride=2, padding_type='SAME'):
    return tf.nn.max_pool(input, ksize=[1, overlap, overlap, 1], strides=[1, stride, stride, 1],
                          padding=padding_type)


def conv_layer(input, weight_shape, bias_shape,
               weight_type='random_normal', bias_type='random_normal',
               stride=2):
    weight = weights(weight_shape, weight_type=weight_type, weight_name='weight')
    bias = biases(bias_shape, bias_type=bias_type, bias_name='bias')
    layer = convolve(input, weight, bias)
    layer = maxpool_layer(layer, stride=stride)
    return layer


def batch_norm_layer():
    return True