import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.misc import toimage
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.python.framework import graph_util
from functools import reduce
from operator import mul
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from library.tf.parameters import Weights, Bias
from library.tf.data_augmentation import flip_left_right, flip_up_down, random_rotate, transpose
from library.tf.layers import conv_layer, maxpool_layer, mlp_layer, dropout, optimize_algo
from library.tf.summary import Summaries
from library.plot_tools import plot
from library.utils import file_utils
import os, glob, time, math
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class CNNClassifier:
    def __init__(self, device='/cpu:0',
                 verbose=True,
                 session_type='default',
                 num_iter=100,
                 err_tolerance=1e-7,
                 train_validate_split=None,
                 display_step=10,
                 learn_step=10,
                 learn_rate_type='constant',
                 learn_rate=0.01,
                 batch_size=64,
                 logs=True,
                 log_dir='./logs/',
                 test_log=False,
                 save_model=True,
                 save_checkpoint=True,
                 checkpoint_filename='cnn_classifier_model.ckpt',
                 model_name='cnn_classifier_model.pb',
                 restore=True,
                 descent_method='gradient',
                 config=None,
                 augmentation=None,
                 ):
        self.verbose = verbose
        # Classifier Parameter configuration
        self.config = config
        self.augmentation = augmentation
        self.descent_method = descent_method
        self.num_classes = None
        self.image_shape = None
        self.num_layers = None
        self.batch_size = batch_size
        self.model_params = dict()  # weights, bias
        self.model_params['weights'] = {}
        self.model_params['bias'] = {}
        self.model_params['layers'] = {}
        self.params = dict()  # any other params
        self.predict_params = dict()  # input, class output, one hot output
        self.out_params = dict()  # loss and accuracy
        self.out_params['list_train_loss'] = []
        self.out_params['list_train_acc'] = []
        self.out_params['list_val_loss'] = []
        self.out_params['list_val_acc'] = []
        self.out_params['list_test_loss'] = []
        self.out_params['list_test_acc'] = []
        self.out_params['list_learn_rate'] = []
        self.summary_params = {}
        # Learning parameters
        self.learn_rate_type = learn_rate_type
        self.learn_rate = learn_rate
        # Logging Parameters
        self.logging = logs
        self.logging_dir = log_dir
        self.summary_writer = None
        self.test_log = test_log
        self.merged_summary_op = None
        # Model Parameters
        self.save_model = save_model
        self.save_checkpoint = save_checkpoint
        self.model = None
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.restore = restore
        self.current_learn_rate = None
        self.last_epoch = None
        self.global_step = None
        # Running configuration
        self.device = device
        self.session_type = session_type
        self.session = None
        self.max_iterations = num_iter
        self.err_tolerance = err_tolerance
        self.train_validate_split = train_validate_split
        if self.train_validate_split is None:
            self.train_validate_split = 0.8
        # External parameters
        self.display_step = display_step
        self.learn_step = learn_step
        self.summary_class = Summaries(train=True, validate=True, test=test_log)

    def print_parameters(self):
        print('Parameters for CNN classifier')
        print('>> Input Parameters')
        print('Input                  : %s ' % str(self.predict_params['input']))
        print('True one hot labels    : %s ' % str(self.predict_params['true_one_hot']))
        print('True class             : %s ' % str(self.predict_params['true_class']))
        print('Predict one hot labels : %s ' % str(self.predict_params['predict_one_hot']))
        print('Predict class          : %s ' % str(self.predict_params['predict_class']))
        print('Output Logits  : %s' % str(self.params['logits']))
        print('Entropy        : %s' % str(self.params['cross_entropy']))
        print('Predictions    : %s ' % str(self.params['predictions']))
        print('Optimizer      : %s ' % str(self.params['optimizer']))
        print('>> Model params')
        for op in self.session.graph.get_operations():
            print(op.name)
        print('>> Model info')
        print('Train loss     : %s' % str(self.model_params['train_loss']))
        print('Train accuracy : %s' % str(self.model_params['train_acc']))
        print('Val. loss      : %s' % str(self.model_params['val_loss']))
        print('Val. accuracy  : %s' % str(self.model_params['val_acc']))
        print('>> Summary params')
        print('Train loss     : %s' % str(self.summary_params['train_loss']))
        print('Train accuracy : %s' % str(self.summary_params['train_acc']))
        print('Val. loss      : %s' % str(self.summary_params['val_loss']))
        print('Val. accuracy  : %s' % str(self.summary_params['val_acc']))

    def make_placeholders_for_inputs(self):
        with tf.device(self.device):
            with tf.name_scope('Inputs'):
                with tf.name_scope('Data'):
                    self.predict_params['input'] = \
                        tf.placeholder(tf.float32,
                                       [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                       name='X_input')
                with tf.name_scope('Train_Labels'):
                    self.predict_params['true_one_hot'] = \
                        tf.placeholder(tf.float32, [None, self.num_classes], name='y_true_one_hot_label')
                    self.predict_params['true_class'] = \
                        tf.placeholder(tf.int64, [None], name='y_true_class')

    def make_conv_layer(self, key, input, filter_size, num_input, num_output):
        with tf.name_scope(key):
            # Weights
            if 'type' in self.config[key]['weight']:
                weight_type = self.config[key]['weight']['type']
            else:
                weight_type = 'random_normal'
            if 'name' in self.config[key]['weight']:
                weight_name = self.config[key]['weight']['name']
            else:
                weight_name = 'Weight_'+key
            weight_shape = (filter_size, filter_size, num_input, num_output)
            print(weight_name)
            self.model_params['weights'][key] = \
                Weights.define(weight_shape, weight_type=weight_type, weight_name=weight_name)
            # Bias
            if 'type' in self.config[key]['bias']:
                bias_type = self.config[key]['bias']['type']
            else:
                bias_type = 'random_normal'
            if 'name' in self.config[key]['bias']:
                bias_name = self.config[key]['bias']['name']
            else:
                bias_name = 'Bias_'+key
            bias_shape = (num_output,)
            self.model_params['bias'][key] = \
                Bias.define(bias_shape, bias_type=bias_type, bias_name=bias_name)
            if 'name' in self.config[key]:
                layer_name = self.config[key]['name']
            else:
                layer_name = key
            if 'padding' in self.config[key]:
                pad = self.config[key]['padding']
            else:
                pad = 'SAME'
            convolution_layer = conv_layer(input, self.model_params['weights'][key],
                                           self.model_params['bias'][key], dim=2,
                                           stride=self.config[key]['stride'],
                                           padding_type=pad,
                                           layer_name=layer_name)
            self.model_params['layers'][key] = convolution_layer

    def make_maxpool_layer(self, key, input):
        with tf.name_scope(key):
            if 'name' in self.config[key]:
                layer_name = self.config[key]['name']
            else:
                layer_name = key
            if 'padding' in self.config[key]:
                pad = self.config[key]['padding']
            else:
                pad = 'SAME'
            max_pool_layer = maxpool_layer(input, dim=2,
                                           stride=self.config[key]['stride'],
                                           padding_type=pad,
                                           overlap=self.config[key]['overlap'],
                                           layer_name=layer_name)
            self.model_params['layers'][key] = max_pool_layer

    def make_dropout(self, input, key):
        self.model_params['layers'][key] = dropout(input, self.config[key]['dropout'])

    def make_full_connected_layer(self, key, input, weight_shape, bias_shape):
        with tf.name_scope(key):
            # Weights
            if 'type' in self.config[key]['weight']:
                weight_type = self.config[key]['weight']['type']
            else:
                weight_type = 'random_normal'
            if 'name' in self.config[key]['weight']:
                weight_name = self.config[key]['weight']['name']
            else:
                weight_name = 'Weight_' + key
            self.model_params['weights'][key] = \
                Weights.define(weight_shape, weight_type=weight_type, weight_name=weight_name)
            # Bias
            if 'type' in self.config[key]['bias']:
                bias_type = self.config[key]['bias']['type']
            else:
                bias_type = 'random_normal'
            if 'name' in self.config[key]['bias']:
                bias_name = self.config[key]['bias']['name']
            else:
                bias_name = 'Bias_' + key
            self.model_params['bias'][key] = \
                Bias.define(bias_shape, bias_type=bias_type, bias_name=bias_name)
            if 'name' in self.config[key]:
                layer_name = self.config[key]['name']
            else:
                layer_name = key
            max_pool_layer = mlp_layer(input, self.model_params['weights'][key],
                                       self.model_params['bias'][key],
                                       layer_name=layer_name)
            self.model_params['layers'][key] = max_pool_layer

    def make_layers(self):
        config_keys = list(self.config)
        iteration = 0
        prev_key = None
        for key in config_keys:
            print('%s -> %s' % (str(prev_key), key))
            if 'conv' in key:
                print('Making convolution layer: %s' % key)
                if iteration == 0:
                    input = self.predict_params['input']
                    num_input = self.image_shape[2]
                else:
                    input = self.model_params['layers'][prev_key]
                    num_input = input.get_shape().as_list()[-1]
                filter_size = self.config[key]['filter_size']
                num_output = self.config[key]['num_outputs']
                print(filter_size, filter_size, num_input, num_output)
                print('Input shape: %s' % str(input.get_shape().as_list()))
                self.make_conv_layer(key, input, filter_size, num_input, num_output)
                print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
            if 'max_pool' in key:
                print('Making max pool layer: %s' % key)
                input = self.model_params['layers'][prev_key]
                print('Input shape: %s' % str(input.get_shape().as_list()))
                self.make_maxpool_layer(key, input)
                print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                if 'dropout' in key:
                    print('Adding dropout')
                    print('Input shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                    self.make_dropout(self.model_params['layers'][key], key)
                    print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
            if 'full_connected' in key:
                print('Making full connected layer: %s' % key)
                input = self.model_params['layers'][prev_key]
                if 'conv' in prev_key or 'max_pool' in prev_key:
                    num_nodes_list = self.model_params['layers'][prev_key].get_shape().as_list()
                    num_nodes = reduce(mul, num_nodes_list[1:], 1)
                    input = tf.reshape(input, [-1, num_nodes])
                else:
                    input = self.model_params['layers'][prev_key]
                    num_nodes = input.get_shape().as_list()[-1]
                weight_shape = (num_nodes, self.config[key]['num_outputs'])
                bias_shape = (self.config[key]['num_outputs'],)
                print('Input shape: %s' % str(input.get_shape().as_list()))
                self.make_full_connected_layer(key, input, weight_shape, bias_shape)
                print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                if 'dropout' in key:
                    print('Adding dropout')
                    print('Input shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                    self.make_dropout(self.model_params['layers'][key], key)
                    print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
            if 'output' in key:
                print('Making output layer: %s' % key)
                num_nodes = self.model_params['layers'][prev_key].get_shape().as_list()[-1]
                input = self.model_params['layers'][prev_key]
                weight_shape = (num_nodes, self.num_classes)
                bias_shape = (self.num_classes,)
                print('Input shape: %s' % str(input.get_shape().as_list()))
                self.make_full_connected_layer(key, input, weight_shape, bias_shape)
                if 'dropout' in key:
                    print('Adding dropout')
                    print('Input shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                    self.make_dropout(self.model_params['layers'][key], key)
                    print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
                print('Output shape: %s' % str(self.model_params['layers'][key].get_shape().as_list()))
            prev_key = key
            iteration += 1

    def make_parameters(self):
        with tf.device(self.device):
            with tf.name_scope('Layers'):
                self.make_layers()

    def make_predictions(self):
        with tf.device(self.device):
            with tf.name_scope('Predictions'):
                self.params['logits'] = tf.nn.softmax(self.model_params['layers']['output_layer'])
                self.predict_params['predict_class'] = \
                    tf.argmax(self.params['logits'], dimension=1, name='predict_class')
                self.predict_params['predict_one_hot'] = \
                    tf.one_hot(self.predict_params['predict_class'],
                               depth=self.num_classes, on_value=1.0,
                               off_value=0.0, axis=-1)

    def make_optimization(self):
        with tf.device(self.device):
            with tf.name_scope('Cross_Entropy'):
                self.params['cross_entropy'] = \
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.model_params['layers']['output_layer'],
                                                            labels=self.predict_params['true_one_hot'])
                print('Made cross entropy')
                nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
                print(nodes)
            with tf.name_scope('Loss_Function'):
                total_loss = tf.reduce_mean(self.params['cross_entropy'])
                # Train loss
                self.model_params['train_loss'], self.params['var_train_loss'], \
                self.params['update_train_loss'], self.summary_params['train_loss'] = \
                    self.summary_class.write_loss(total_loss, self.out_params['list_train_loss'],
                                                  name='train')
                # Validation loss
                self.model_params['val_loss'], self.params['var_val_loss'], \
                self.params['update_val_loss'], self.summary_params['val_loss'] = \
                    self.summary_class.write_loss(total_loss, self.out_params['list_val_loss'],
                                                  name='validate')
        with tf.device(self.device):
            with tf.name_scope('Optimizer'):
                if self.learn_rate_type == 'exponential':
                    self.current_learn_rate = \
                        tf.train.exponential_decay(self.learn_rate, self.global_step,
                                                   self.display_step, 0.96,
                                                   staircase=True)
                else:
                    self.current_learn_rate = self.learn_rate
                self.params['var_learn_rate'], self.params['update_learn_rate'], \
                self.summary_params['learn_rate'] = \
                    self.summary_class.write_learn_rate(self.current_learn_rate,
                                                        self.out_params['list_learn_rate'])
                self.params['optimizer'] = \
                    optimize_algo(self.current_learn_rate,
                                  descent_method=self.descent_method) \
                        .minimize(self.model_params['train_loss'])
        with tf.device(self.device):
            self.params['predictions'] = tf.equal(self.predict_params['true_class'],
                                                  self.predict_params['predict_class'])

    def make_accuracy(self):
        with tf.device(self.device):
            with tf.name_scope('Accuracy'):
                self.model_params['train_acc'], self.params['var_train_acc'], \
                self.params['update_train_acc'], self.summary_params['train_acc'] = \
                    self.summary_class.write_accuracy(self.params['predictions'],
                                                      self.out_params['list_train_acc'],
                                                      name='train')
                self.model_params['val_acc'], self.params['var_val_acc'], \
                self.params['update_val_acc'], self.summary_params['val_acc'] = \
                    self.summary_class.write_accuracy(self.params['predictions'],
                                                      self.out_params['list_val_acc'],
                                                      name='validate')
                if self.test_log is True:
                    self.model_params['test_acc'], self.params['var_test_acc'], \
                    self.params['update_test_acc'], self.summary_params['test_acc'] = \
                        self.summary_class.write_accuracy(self.params['predictions'],
                                                          self.out_params['list_test_acc'],
                                                          name='test')

    def create_graph(self, image_shape, num_classes):
        start = time.time()
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_layers = len(self.config.keys())-1
        self.global_step = tf.Variable(0, name='last_successful_epoch', trainable=False, dtype=tf.int32)
        self.last_epoch = tf.assign(self.global_step, self.global_step + 1, name='assign_updated_epoch')
        # Step 1: Creating placeholders for inputs
        self.make_placeholders_for_inputs()
        # Step 2: Creating initial parameters for the variables
        self.make_parameters()
        # Step 3: Make predictions for the data
        self.make_predictions()
        # Step 4: Perform optimization operation
        self.make_optimization()
        # Step 5: Calculate accuracies
        self.make_accuracy()
        # Step 6: Initialize all the required variables
        with tf.device(self.device):
            self.init_var = tf.global_variables_initializer()
        # Step 7: Initiate Session
        config = tf.ConfigProto(
            log_device_placement=True,
            allow_soft_placement=True,
        )
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        if self.session_type == 'default':
            self.session = tf.Session(config=config)
        if self.session_type == 'interactive':
            self.session = tf.InteractiveSession(config=config)
        print('Session: ' + str(self.session))
        self.session.run(self.init_var)
        # Step 8: Initiate logs
        if self.logging is True:
            file_utils.mkdir_p(self.logging_dir)
            self.merged_summary_op = tf.summary.merge_all()
            if self.restore is False:
                file_utils.delete_all_files_in_dir(self.logging_dir)
            self.summary_writer = \
                tf.summary.FileWriter(self.logging_dir, graph=self.session.graph)
            if self.save_checkpoint is True:
                self.model = tf.train.Saver(max_to_keep=1)
        # Step 9: Restore model
        if self.restore is True:
            self.restore_model()
        epoch = self.session.run(self.global_step)
        print('Model has been trained for %d iterations' % epoch)
        end = time.time()
        print('Tensorflow graph created in %.4f seconds' % (end - start))
        return True

    def make_data_augmentation(self, data, labels, classes):
        aug_keys = list(self.augmentation)
        aug_data = []
        aug_labels = []
        aug_classes = []
        result_data = None
        result_labels = None
        result_classes = None
        iteration = 0
        for key in aug_keys:
            if 'flip_left_right' in key:
                if 'seed' not in self.augmentation[key].keys():
                    seed = None
                else:
                    seed = self.augmentation[key]['seed']
                result_data, result_labels, result_classes = \
                    flip_left_right(data, labels, classes, random_seed=seed)
            elif 'flip_up_down' in key:
                result_data, result_labels, result_classes = \
                    flip_up_down(data, labels, classes)
                result_data = self.session.run(result_data)
            elif 'rotate' in key:
                if 'seed' not in self.augmentation[key].keys():
                    seed = None
                else:
                    seed = self.augmentation[key]['seed']
                if 'max_angle' not in self.augmentation[key].keys():
                    max_angle = None
                else:
                    max_angle = self.augmentation[key]['max_angle']
                result_data, result_labels, result_classes = \
                    random_rotate(data, labels, classes, random_seed=seed,
                                  max_angle=max_angle)
            if iteration == 0:
                aug_data = result_data
                aug_labels = result_labels
                aug_classes = result_classes
            else:
                aug_data = np.append(aug_data, result_data, axis=0)
                aug_labels = np.append(aug_labels, result_labels, axis=0)
                aug_classes = np.append(aug_classes, result_classes, axis=0)
            iteration += 1
        del result_data
        del result_labels
        del result_classes
        return aug_data, aug_labels, aug_classes

    def make_data(self, data, labels, classes):
        if self.augmentation is not None:
            print('Performing augmentation')
            data, labels, classes = self.make_data_augmentation(data, labels, classes)
        return data, labels, classes

    def fit(self, data, labels, classes, test_data=None, test_labels=None, test_classes=None):
        # if self.augmentation is not None:
        #     aug_data, aug_labels, aug_classes = \
        #         self.make_data_augmentation(data, labels, classes)
        # data = np.append(data, aug_data, axis=0)
        # labels = np.append(labels, aug_labels, axis=0)
        # classes = np.append(classes, aug_classes, axis=0)
        train_data, validate_data, train_labels, \
        validate_labels, train_classes, validate_classes = \
            train_test_split(data, labels, classes, train_size=self.train_validate_split)
        if self.verbose is True:
            print('Data shape             : ' + str(data.shape))
            print('Labels shape           : ' + str(labels.shape))
            print('Classes shape          : ' + str(classes.shape))
            print('Train Data shape       : ' + str(train_data.shape))
            print('Train Labels shape     : ' + str(train_labels.shape))
            print('Train Classes shape    : ' + str(train_classes.shape))
            print('Validate Data shape    : ' + str(validate_data.shape))
            print('Validate Labels shape  : ' + str(validate_labels.shape))
            print('Validate Classes shape : ' + str(validate_classes.shape))
        if self.test_log is False:
            self.learn(train_data, train_labels, train_classes,
                       validate_data=validate_data, validate_labels=validate_labels,
                       validate_classes=validate_classes)
        else:
            self.learn(train_data, train_labels, train_classes,
                       validate_data=validate_data, validate_labels=validate_labels,
                       validate_classes=validate_classes, test_data=test_data,
                       test_labels=test_labels, test_classes=test_classes)

    def learn(self, train_data, train_labels, train_classes,
              validate_data=None, validate_labels=None, validate_classes=None,
              test_data=None, test_labels=None, test_classes=None):
        start = time.time()
        feed_dict_validate = {self.predict_params['input']: validate_data,
                              self.predict_params['true_one_hot']: validate_labels,
                              self.predict_params['true_class']: validate_classes}
        epoch = self.session.run(self.global_step)
        self.out_params['list_train_loss'] = self.session.run(self.params['var_train_loss']).tolist()
        self.out_params['list_train_acc'] = self.session.run(self.params['var_train_acc']).tolist()
        print('Length of train loss          : %d' % len(self.out_params['list_train_loss']))
        print('Length of train accuracy      : %d' % len(self.out_params['list_train_acc']))
        self.out_params['list_val_loss'] = self.session.run(self.params['var_val_loss']).tolist()
        self.out_params['list_val_acc'] = self.session.run(self.params['var_val_acc']).tolist()
        print('Length of validate loss       : %d' % len(self.out_params['list_val_loss']))
        print('Length of validate accuracy   : %d' % len(self.out_params['list_val_acc']))
        if self.test_log is True:
            self.out_params['list_test_acc'] = \
                self.session.run(self.params['var_test_acc']).tolist()
        print('Length of test accuracy       : %d' % len(self.out_params['list_test_acc']))
        print('Restoring training from epoch :', epoch)
        converged = False
        prev_cost = 0
        while (epoch != self.max_iterations) and converged is False:
            start = time.time()
            num_batches = int(math.ceil(train_data.shape[0] / self.batch_size))
            print('Training using original train data using batch size of %d '
                  'and total batches of %d' % (self.batch_size, num_batches))
            start_batch_index = 0
            for batch in range(num_batches):
                end_batch_index = start_batch_index + self.batch_size
                if end_batch_index < train_data.shape[0]:
                    train_batch_data = train_data[start_batch_index:end_batch_index, :]
                    train_batch_labels = train_labels[start_batch_index:end_batch_index, :]
                    train_batch_classes = train_classes[start_batch_index:end_batch_index]
                else:
                    train_batch_data = train_data[start_batch_index:, :]
                    train_batch_labels = train_labels[start_batch_index:, :]
                    train_batch_classes = train_classes[start_batch_index:]
                feed_dict_train = {self.predict_params['input']: train_batch_data,
                                   self.predict_params['true_one_hot']: train_batch_labels,
                                   self.predict_params['true_class']: train_batch_classes}
                _, train_loss, train_loss_summary, \
                train_acc, train_acc_summary, curr_epoch \
                    = self.session.run([self.params['optimizer'],
                                        self.model_params['train_loss'], self.summary_params['train_loss'],
                                        self.model_params['train_acc'], self.summary_params['train_acc'],
                                        self.last_epoch],
                                       feed_dict=feed_dict_train)
                start_batch_index += self.batch_size
            if self.augmentation is not None:
                aug_data, aug_labels, aug_classes = \
                    self.make_data_augmentation(train_data, train_labels, train_classes)
                num_batches = int(math.ceil(aug_data.shape[0] / self.batch_size))
                print('Training using augmented train data using batch size of %d '
                      'and total batches of %d' % (self.batch_size, num_batches))
                start_batch_index = 0
                for batch in range(num_batches):
                    end_batch_index = start_batch_index + self.batch_size
                    if end_batch_index < aug_data.shape[0]:
                        aug_batch_data = aug_data[start_batch_index:end_batch_index, :]
                        aug_batch_labels = aug_labels[start_batch_index:end_batch_index, :]
                        aug_batch_classes = aug_classes[start_batch_index:end_batch_index]
                    else:
                        aug_batch_data = aug_data[start_batch_index:, :]
                        aug_batch_labels = aug_labels[start_batch_index:, :]
                        aug_batch_classes = aug_classes[start_batch_index:]
                    feed_dict_train = {self.predict_params['input']: aug_batch_data,
                                       self.predict_params['true_one_hot']: aug_batch_labels,
                                       self.predict_params['true_class']: aug_batch_classes}
                    _, train_loss, train_loss_summary, \
                    train_acc, train_acc_summary, curr_epoch \
                        = self.session.run([self.params['optimizer'],
                                            self.model_params['train_loss'], self.summary_params['train_loss'],
                                            self.model_params['train_acc'], self.summary_params['train_acc'],
                                            self.last_epoch],
                                           feed_dict=feed_dict_train)
                    start_batch_index += self.batch_size
            val_loss, val_loss_summary, val_acc, val_acc_summary = \
                self.session.run([self.model_params['val_loss'], self.summary_params['val_loss'],
                                  self.model_params['val_acc'], self.summary_params['val_acc']],
                                 feed_dict=feed_dict_validate)
            learn_rate_summary = self.session.run(self.summary_params['learn_rate'])
            self.out_params['list_train_loss'].append(train_loss)
            self.params['update_train_loss'] = tf.assign(self.params['var_train_loss'],
                                                         self.out_params['list_train_loss'],
                                                         validate_shape=False).eval()
            self.out_params['list_train_acc'].append(train_acc)
            self.params['update_train_acc'] = \
                tf.assign(self.params['var_train_acc'], self.out_params['list_train_acc'],
                          validate_shape=False).eval()
            self.out_params['list_learn_rate'].append(self.current_learn_rate)
            self.params['update_learn_rate'] = \
                tf.assign(self.params['var_learn_rate'], self.out_params['list_learn_rate'],
                          validate_shape=False).eval()
            self.out_params['list_val_loss'].append(val_loss)
            self.params['update_val_loss'] = \
                tf.assign(self.params['var_val_loss'], self.out_params['list_val_loss'],
                          validate_shape=False).eval()
            self.out_params['list_val_acc'].append(val_acc)
            self.params['update_val_acc'] = \
                tf.assign(self.params['var_val_acc'], self.out_params['list_val_acc'],
                          validate_shape=False).eval()
            self.summary_writer.add_summary(train_loss_summary, epoch)
            self.summary_writer.add_summary(train_acc_summary, epoch)
            self.summary_writer.add_summary(val_loss_summary, epoch)
            self.summary_writer.add_summary(val_acc_summary, epoch)
            self.summary_writer.add_summary(learn_rate_summary, epoch)
            if self.test_log is True:
                feed_dict_test = {self.predict_params['input']: test_data,
                                  self.predict_params['true_one_hot']: test_labels,
                                  self.predict_params['true_class']: test_classes}
                test_acc, test_acc_summary = \
                    self.session.run([self.model_params['test_acc'],
                                      self.summary_params['test_acc']], feed_dict=feed_dict_test)
                self.out_params['list_test_acc'].append(test_acc)
                self.params['update_test_acc'] = tf.assign(self.params['var_test_acc'],
                                                           self.out_params['list_test_acc'],
                                                           validate_shape=False).eval()
                self.summary_writer.add_summary(test_acc_summary, epoch)
            if epoch % self.display_step == 0:
                duration = time.time() - start
                if self.test_log is False:
                    print('>>> Epoch [%*d/%*d]'
                          % (int(len(str(self.max_iterations))), epoch,
                             int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | val_loss: %.4f | val_acc: %.4f | '
                          'Time: %.4f s' % (train_loss, train_acc, val_loss, val_acc, duration))
                else:
                    print('>>> Epoch [%*d/%*d]'
                          % (int(len(str(self.max_iterations))), epoch,
                             int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | val_loss: %.4f | val_acc: %.4f | '
                          'test_acc: %.4f | Time: %.4f s'
                          % (train_loss, train_acc, val_loss, val_acc, test_acc, duration))
            if self.save_checkpoint is True:
                model_directory = os.path.dirname(self.checkpoint_filename)
                file_utils.mkdir_p(model_directory)
                self.model.save(self.session, self.checkpoint_filename, global_step=epoch)
            if epoch == 0:
                prev_cost = train_loss
            else:
                if math.fabs(train_loss - prev_cost) < self.err_tolerance:
                    converged = False
            epoch += 1
        end = time.time()
        print('Fit completed in %.4f seconds' % (end - start))
        if self.save_model is True:
            print('Saving the graph to %s' % (self.logging_dir+self.model_name.split('/')[-1]))
            self.freeze_graph(self.logging_dir)

    def predict(self, data):
        feed_dict_data = {self.predict_params['input']: data}
        predictions = self.session.run(self.predict_params['predict_class'],
                                       feed_dict=feed_dict_data)
        predictions = np.array(predictions)
        return predictions

    def print_classification_results(self, test_data,
                                     test_labels,
                                     test_classes,
                                     test_class_names=[],
                                     normalize=True):
        feed_dict_test = {self.predict_params['input']: test_data,
                          self.predict_params['true_one_hot']: test_labels,
                          self.predict_params['true_class']: test_classes}
        if len(test_class_names) == 0:
            unique_values = list(set(test_classes))
            num_classes = len(unique_values)
            test_class_names = []
            for classes in range(num_classes):
                test_class_names.append('Class ' + str(classes))
        cls_true = test_classes
        cls_pred = self.session.run(self.predict_params['predict_class'],
                                    feed_dict=feed_dict_test)
        plot.plot_confusion_matrix(cls_true, cls_pred, classes=test_class_names,
                                   normalize=normalize, title='Confusion matrix')
        print('Detailed classification report')
        print(classification_report(y_true=cls_true, y_pred=cls_pred,
                                    target_names=test_class_names))

    def score(self, test_data, test_classes):
        predict_classes = self.predict(test_data)
        return accuracy_score(test_classes, predict_classes, normalize=True)

    def plot_loss(self):
        loss = np.vstack((self.out_params['list_train_loss'],
                          self.out_params['list_val_loss']))
        plot.plot_scores(loss, legend=['train_loss', 'validate_loss'],
                         colors=['blue', 'green'], plot_title='Loss of linear classifier',
                         plot_xlabel='No. of iterations', plot_ylabel='Loss',
                         plot_lib='matplotlib', matplotlib_style='default')
        return True

    def plot_accuracy(self):
        accuracy = np.vstack((self.out_params['list_train_acc'],
                              self.out_params['list_val_acc']))
        plot.plot_scores(accuracy, legend=['train_accuracy', 'validate_accuarcy'],
                         colors=['blue', 'green'], plot_title='Accuarcy of linear classifier',
                         plot_xlabel='No. of iterations', plot_ylabel='Accuracy',
                         plot_lib='matplotlib', matplotlib_style='default')
        return True

    def plot_layers(self, input_images, layer_key, num_layers=32, type='rgb', fig_size=(12,16), fontsize=20):
        feed_dict_layer = {self.predict_params['input']: input_images}
        layer = self.session.run(self.model_params['layers'][layer_key], feed_dict=feed_dict_layer)
        layer = layer[0, :]
        print(layer.shape)
        if layer.shape[-1] < num_layers:
            # print('Requested more layers than present. Reducing to %d' % layer.shape[-1])
            num_layers = layer.shape[-1]
        num_rows = 4
        num_cols = int(num_layers / 4)
        fig = plt.figure()
        fig.set_figheight(fig_size[0])
        fig.set_figwidth(fig_size[1])
        fig.subplots_adjust(wspace=0.1, hspace=0.0001)
        gs = gridspec.GridSpec(num_rows, num_cols)
        for layer_no in range(num_layers):
            ax = plt.subplot(gs[layer_no])
            ax.set_aspect('equal')
            if type == 'rgb':
                ax.imshow(toimage(layer[:, :, layer_no]), cmap='binary')
            elif type == 'grey':
                ax.imshow(toimage(layer[:, :, layer_no]), cmap=matplotlib.cm.Greys_r)
            else:
                ax.imshow(layer[:, layer_no], cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
        fig_title = 'Convolution layer: ' + layer_key
        fig.suptitle(fig_title, fontsize=fontsize)
        plt.show()
        return True

    def restore_model(self):
        file_name = os.path.splitext(os.path.abspath(self.model_name))[0]
        num_files = len(sorted(glob.glob(os.path.abspath(file_name + '*.meta'))))
        if num_files > 0:
            checkpoint_file = os.path.abspath(sorted(glob.glob(file_name + '*.data-00000-of-00001'),
                                                     reverse=True)[0])
            if os.path.exists(checkpoint_file):
                print('Restoring model from %s' % checkpoint_file)
                meta_file = os.path.abspath(sorted(glob.glob(file_name + '*.meta'), reverse=True)[0])
                print('Loading: %s' % meta_file)
                saver = tf.train.import_meta_graph(meta_file)
                print('Loading: %s' % os.path.abspath(checkpoint_file))
                cpk = tf.train.latest_checkpoint(os.path.dirname(meta_file))
                print('Checkpoint: ' + str(cpk))
                print('Tensors')
                print(print_tensors_in_checkpoint_file(file_name=cpk, all_tensors='', tensor_name=''))
                saver.restore(self.session, tf.train.latest_checkpoint(os.path.dirname(meta_file)))
                print('Last epoch to restore: ' + str(self.session.run(self.global_step)))
        else:
            file_utils.delete_all_files_in_dir(self.logging_dir)
            print('Restoring cannot be done')

    def freeze_graph(self, model_folder):
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print(input_checkpoint)
        absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
        print(absolute_model_folder)
        output_graph = absolute_model_folder + '/' + self.model_name.split('/')[-1]
        print(output_graph)
        output_node_names = 'Predictions/predict_classes'
        clear_devices = True
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        print(saver)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(',')
            )
            with tf.gfile.GFile(output_graph, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))
        return True

    def load_model(self, model_name):
        self.model.restore(self.session, model_name)

    def close(self):
        self.session.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.summary_writer.close()
        self.session.close()

    def __del__(self):
        self.summary_writer.close()
        self.session.close()
