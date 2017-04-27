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
    elif activation_type == 'relu':
        layer = tf.nn.relu(input, name=activation_name)
    elif activation_type == 'tanh':
        layer = tf.nn.tanh(input, name=activation_name)
    else:
        layer = tf.nn.relu(input, name=activation_name)
    return layer


def mlp_layer(prev_layer, weight, bias, activation_type='sigmoid',
              layer_name='mlp_layer'):
    var = tf.nn.xw_plus_b(prev_layer, weight, bias, name=layer_name)
    if activation_type != '':
        layer = activation_layer(var, activation_type=activation_type,
                                 activation_name=layer_name)
    else:
        layer = var
    return layer


def optimize_algo(learning_rate=0.001, descent_method='gradient',
                  adam_beta1=0.9, adam_beta2=0.999,
                  momentum=0.9, lr_decay=0.0, decay_step=100):
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
    elif descent_method == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return optimizer


def convolve(input, weight, bias, stride=1, activation_type='relu', dim=2,
             padding_type='SAME', conv_layer_name=''):
    if dim == 2:
        x = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1],
                         padding=padding_type)
    elif dim == 3:
        x = tf.nn.conv3d(input, weight, strides=[1, stride, stride, stride, 1],
                         padding=padding_type)
    else:
        raise ValueError('Dim cannot be other than 2 or 3')
    x = tf.nn.bias_add(x, bias)
    return activation_layer(x, activation_type=activation_type, activation_name=conv_layer_name)


def maxpool_layer(input, overlap=2, stride=2, dim=2, padding_type='SAME',
                  layer_name=''):
    if dim == 2:
        x = tf.nn.max_pool(input, ksize=[1, overlap, overlap, 1],
                           strides=[1, stride, stride, 1],
                           padding=padding_type, name=layer_name)
    elif dim == 3:
        x = tf.nn.max_pool3d(input, ksize=[1, overlap, overlap, overlap, 1],
                             strides=[1, stride, stride, stride, 1],
                             padding=padding_type, name=layer_name)
    else:
        raise ValueError('Dim cannot be other than 2 or 3')
    return x


def conv_layer(input, weight, bias, stride=2, dim=2, padding_type='SAME', layer_name=''):
    layer = convolve(input, weight, bias, stride=stride, dim=dim, padding_type=padding_type,
                     conv_layer_name=layer_name)
    return layer


def dropout(input, dropout=0.8, layer_name=''):
    return tf.nn.dropout(input, dropout, name=layer_name)


def batch_norm_layer(input):
    return tf.nn.batch_normalization(input)


def residual_layer(input):
    return True