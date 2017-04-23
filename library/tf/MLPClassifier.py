import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from library.utils import file_utils
from library.tf import Layers
from library.plot_tools import plot_tools
from library.preprocessing import data_transform
import math
import os, glob, time, re
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# Resources
# https://www.youtube.com/watch?v=3VEXX73tnw4

# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(3072, 3072), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=1000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
#        warm_start=False)


class TFMLPClassifier:

    def __init__(self, session_type='default', save_model=False, restore=True, device='',
                 logs=True, log_dir='./logs/', model_name='./model/linear_classifier_model.ckpt', test_log=True,
                 display_step=10, learning_rate=0.01, learning_rate_type='constant',
                 descent_method='gradient', adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                 reg_const=0.0001, regularize=False,
                 nodes_in_layers=[5, 5], init_weights=['random'], init_bias=['ones'], activation_req=False,
                 activation_fn=['softmax'], batch_size=100, tolerance=1e-7, num_iterations=100,
                 transform=True, transform_method='StandardScaler',
                 train_validate_split=None, separate_writer=False,
                 verbose=False):
        self.verbose = verbose
        self.restore = restore
        # Tensorflow logs and models
        self.tensorboard_log_dir = log_dir
        self.tensorboard_logs = logs
        self.merged_summary_op = None
        # Log writers
        self.separate_writer = separate_writer
        self.summary_writer = None
        self.train_writer = None
        self.validate_writer = None
        self.test_writer = None
        # Model info
        self.model = None
        self.model_name = model_name
        self.save_model = save_model
        # Summary methods
        self.train_loss_summary = None
        self.train_acc_summary = None
        self.validate_loss_summary = None
        self.validate_acc_summary = None
        self.learning_rate_summary = None
        self.weights_hist_summary = None
        self.bias_hist_summary = None
        self.test_acc_summary = None
        self.w_hist = None
        self.w_im = None
        self.b_hist = None
        # Tensorflow variables for later use
        self.var_train_loss = None
        self.update_train_loss = None
        self.list_train_loss = []
        self.var_train_acc = None
        self.update_train_acc = None
        self.list_train_acc = []
        self.var_validate_loss = None
        self.update_validate_loss = None
        self.list_validate_loss = []
        self.var_validate_acc = None
        self.update_validate_acc = None
        self.list_validate_acc = []
        self.test_var_acc = None
        self.update_test_acc = None
        self.list_test_acc = []
        self.var_learning_rate = None
        self.update_learning_rate = None
        self.list_learning_rate = []
        #
        self.session = None
        if device == '' or device is None:
            self.device = '/cpu:0'
        else:
            self.device = device
        self.session_type = session_type
        # Parameters
        self.learning_rate_type = learning_rate_type
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.max_iterations = num_iterations
        self.display_step = display_step
        self.tolerance = tolerance
        self.descent_method = descent_method
        self.hidden_layers = nodes_in_layers
        self.activation_req = activation_req
        if len(self.hidden_layers) > len(init_weights):
            if len(init_weights) > 0:
                init_weights.extend([init_weights[0]] * (len(self.hidden_layers) - len(init_weights)))
                self.init_weights = init_weights
            else:
                self.init_weights = init_weights[0] * len(init_weights)
        else:
            self.init_weights = init_weights
        if len(self.hidden_layers) > len(init_bias):
            if len(init_bias) > 0:
                init_bias.extend([init_bias[0]] * (len(self.hidden_layers) - len(init_bias)))
                self.init_bias = init_bias
            else:
                self.init_bias = init_bias[0] * len(init_bias)
        else:
            self.init_bias = init_bias
        if len(self.hidden_layers) > len(activation_fn):
            if len(activation_fn) > 0:
                activation_fn.extend([activation_fn[0]] * (len(self.hidden_layers) - len(activation_fn)))
                self.activation_fn = activation_fn
            else:
                self.activation_fn = activation_fn[0] * len(activation_fn)
        else:
            self.activation_fn = activation_fn
        self.regularize = regularize
        if len(self.hidden_layers) > len(reg_const):
            if len(reg_const) > 0:
                reg_const.extend([reg_const[0]] * (len(self.hidden_layers) - len(reg_const)))
                self.reg_const = reg_const
            else:
                self.reg_const = reg_const[0] * len(reg_const)
        else:
            self.reg_const = reg_const
        self.batch_size = batch_size
        # Data transform methods
        self.transform = transform
        self.transform_method = transform_method
        # Graph inputs
        self.x = None
        self.y_true = None
        self.y_true_cls = None
        self.num_features = None
        self.num_classes = None
        # Validation and testing
        self.y_pred = None
        self.y_pred_cls = None
        #
        self.init_var = None
        self.last_epoch = 0
        self.global_step = 0
        self.optimizer = None
        self.train_loss = None
        self.validate_loss = None
        self.train_accuracy = None
        self.validate_accuracy = None
        self.test_accuracy = None
        self.test_log = test_log
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.output_layer = None
        self.correct_prediction = None
        self.cross_entropy = None
        #
        self.train_validate_split = train_validate_split

    def print_parameters(self):
        print('>> Parameters for Linear Classifier')
        print('Activation function        : ', self.activation_fn)
        print('Gradient Descent Method    : ', self.descent_method)
        print('Learning rate type         : ', self.learning_rate_type)
        print('Learning rate              : ', self.learning_rate)
        if self.regularize is True:
            print('Regularization constant    : ', self.reg_const)
        print('Error Tolerance            : ', self.tolerance)
        print('Data Transformation method : ', self.transform_method)
        print('>> Inputs for Tensorflow Graph')
        print('X                          : ', self.x)
        print('Y_true                     : ', self.y_true)
        print('Y_true_cls                 : ', self.y_true_cls)
        print('Device to use              : ', self.device)
        print('>> Output parameters for Tensorflow Graph')
        print('W                          : ' + str(self.weights))
        print('Init Weights               : ' + str(self.init_weights))
        for i in range(len(self.hidden_layers)):
            weight_key = 'weight_' + str(i)
            print('  weight_%d                : %s' % (i, str(self.weights[weight_key])))
        print('b                          : ' + str(self.biases))
        print('Init Bias                  : ' + str(self.init_bias))
        for i in range(len(self.hidden_layers)):
            bias_key = 'bias_' + str(i)
            print('  bias_%d                  : %s' % (i, str(self.biases[bias_key])))
        print('Nodes in layers                : ' + str(self.hidden_layers))
        print('Layers                     : ' + str(self.layers))
        for i in range(len(self.hidden_layers)):
            layer_key = 'layer_' + str(i)
            print('  layer_%d                 : %s' % (i, str(self.layers[layer_key])))
        if self.activation_req is True:
            print('Activation fns.            : ', self.activation_fn)
        print('Output layer               : ' + str(self.output_layer))
        print('Y_pred                     : ', self.y_pred)
        print('Y_pred_cls                 : ', self.y_pred_cls)
        print('cross_entropy              : ', self.cross_entropy)
        print('train_loss                 : ', self.train_loss)
        print('optimizer                  : ', self.optimizer)
        print('correct_prediction         : ', self.correct_prediction)
        print('>> Accuracy parameters')
        print('Train Accuracy             : ', self.train_accuracy)
        print('Validate Accuracy          : ', self.validate_accuracy)
        print('Test Accuracy              : ', self.test_accuracy)

    def make_placeholders_for_inputs(self, num_features, num_classes):
        with tf.device(self.device):
            with tf.name_scope('Inputs'):
                with tf.name_scope('Data'):
                    self.x = tf.placeholder(tf.float32, [None, num_features], name='X')
                with tf.name_scope('Train_Labels'):
                    self.y_true = tf.placeholder(tf.float32, [None, num_classes], name='y_label')
                    self.y_true_cls = tf.placeholder(tf.int64, [None], name='y_class')
                with tf.name_scope('Input_Image'):
                    image_shaped_input = tf.reshape(self.x, [-1, 32, 32, 3])
                    tf.summary.image('Training_Images', image_shaped_input, 1)

    def make_weights(self, num_features, num_classes, number_of_layers=1):
        prev_layer_weights = num_features
        for layer_no in range(number_of_layers):
            weight_shape = (prev_layer_weights, self.hidden_layers[layer_no])
            weight_type = self.init_weights[layer_no]
            weight_key = 'weight_' + str(layer_no)
            weight_name = 'W_'+str(layer_no)+'_'+weight_type
            self.weights[weight_key] = Layers.weights(weight_shape, weight_type=weight_type, weight_name=weight_name)
            # w_h = tf.histogram_summary(weight_type, self.weights[weight_key])
            prev_layer_weights = self.hidden_layers[layer_no]

    def make_bias(self, num_features, num_classes, number_of_layers=1):
        for layer_no in range(number_of_layers):
            bias_shape = [self.hidden_layers[layer_no]]
            bias_type = self.init_bias[layer_no]
            bias_key = 'bias_' + str(layer_no)
            bias_name = 'b_'+str(layer_no)+'_'+bias_type
            self.biases[bias_key] = Layers.biases(bias_shape, bias_type=bias_type, bias_name=bias_name)
            # b_h = tf.histogram_summary(bias_key, self.biases[bias_key])

    def make_layers(self, number_of_layers):
        prev_layer = self.x
        for layer_no in range(number_of_layers):
            layer = None
            weight_key = 'weight_' + str(layer_no)
            bias_key = 'bias_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            layer_type = self.activation_fn[layer_no]
            layer_name = 'Layer_'+str(layer_no)+'_'+layer_type
            logit = tf.add(tf.matmul(prev_layer, self.weights[weight_key]), self.biases[bias_key])
            self.layers[layer_key] = Layers.activation_layer(logit, activation_type=self.activation_fn[layer_no],
                                                             activation_name=layer_name)
            prev_layer = self.layers[layer_key]

    def make_output_layer(self):
        layer_key = layer_key = 'layer_' + str(len(self.hidden_layers)-1)
        output = tf.Variable( tf.random_normal([self.hidden_layers[-1], self.num_classes]))
        bias_output = tf.Variable(tf.random_normal([self.num_classes]))
        self.output_layer = tf.add(tf.matmul(self.layers[layer_key], output), bias_output, name='out_layer')

    def make_parameters(self, num_features, num_classes):
        with tf.device(self.device):
            number_of_layers = len(self.hidden_layers)
            with tf.name_scope('Parameters'):
                with tf.name_scope('Weights'):
                    self.make_weights(num_features, num_classes, number_of_layers)
                with tf.name_scope('Bias'):
                    self.make_bias(num_features, num_classes, number_of_layers)
            with tf.name_scope('Hidden_Layers'):
                self.make_layers(number_of_layers)
            with tf.name_scope('Output_Layer'):
                self.make_output_layer()

    def make_predictions(self):
        with tf.device(self.device):
            with tf.name_scope('Predictions'):
                if self.activation_req is True:
                    self.y_pred = Layers.activation_layer(self.output_layer,
                                                          activation_type=self.activation_fn[-1])
                else:
                    self.y_pred = self.output_layer
                self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

    def make_optimization(self):
        with tf.device(self.device):
            with tf.name_scope('Cross_Entropy'):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer,
                                                                             labels=self.y_true)
            with tf.name_scope('Loss_Function'):
                if self.regularize is True:
                    # ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                    # ridge_loss = tf.reduce_mean(tf.square(self.weights))
                    ridge_param = tf.cast(tf.constant(self.reg_const[0]), dtype=tf.float32)
                    ridge_loss = tf.multiply(ridge_param, tf.reduce_mean(tf.square(self.weights['weight_0'])))
                    for layer_no in range(len(self.hidden_layers)-1):
                        ridge_param = tf.cast(tf.constant(self.reg_const[layer_no+1]), dtype=tf.float32)
                        ridge_loss = tf.add(ridge_loss, tf.multiply(ridge_param,
                                            tf.reduce_mean(tf.square(self.weights['weight_'+str(layer_no+1)]))))
                    self.train_loss = tf.add(tf.reduce_mean(self.cross_entropy), ridge_loss)
                    if self.train_validate_split is not None:
                        self.validate_loss = tf.add(tf.reduce_mean(self.cross_entropy), ridge_loss)
                else:
                    self.train_loss = tf.reduce_mean(self.cross_entropy)
                    if self.train_validate_split is not None:
                        self.validate_loss = tf.reduce_mean(self.cross_entropy)
                self.train_loss_summary = tf.summary.scalar('Training_Error', self.train_loss)
                if self.train_validate_split is not None:
                    self.validate_loss_summary = tf.summary.scalar('Validation_Error', self.validate_loss)
                self.var_train_loss = tf.Variable([], dtype=tf.float32, trainable=False,
                                                    validate_shape=False, name='train_loss_list')
                self.update_train_loss = tf.assign(self.var_train_loss, self.list_train_acc, validate_shape=False)
                if self.train_validate_split is not None:
                    self.var_validate_loss = tf.Variable([], dtype=tf.float32, trainable=False,
                                                      validate_shape=False, name='validate_loss_list')
                    self.update_validate_loss = tf.assign(self.var_validate_loss, self.list_validate_acc, validate_shape=False)
            with tf.name_scope('Optimizer'):
                self.var_learning_rate= tf.Variable([], dtype=tf.float32, trainable=False,
                                                    validate_shape=False, name='learning_rate_progress')
                self.update_learning_rate = tf.assign(self.var_learning_rate, self.list_learning_rate,
                                                      validate_shape=False)
                if self.learning_rate_type == 'exponential':
                    self.current_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                               self.display_step, 0.96, staircase=True)
                else:
                    self.current_learning_rate = self.learning_rate
                self.learning_rate_summary = tf.summary.scalar('Learning_rate', self.current_learning_rate)
                optimizer = Layers.optimize_algo(self.current_learning_rate, descent_method=self.descent_method)
                self.optimizer = optimizer.minimize(self.train_loss)
            self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

    def make_accuracy(self):
        with tf.device(self.device):
            with tf.name_scope('Accuracy'):
                self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                self.var_train_acc = tf.Variable([], dtype=tf.float32, trainable=False,
                                                 validate_shape=False, name='train_accuracy_list')
                self.update_train_acc = tf.assign(self.var_train_acc, self.list_train_acc,
                                                  validate_shape=False)
                if self.separate_writer is True:
                    self.train_acc_summary = tf.summary.scalar('Train_Accuracy', self.train_accuracy)
                else:
                    self.train_acc_summary = tf.summary.scalar('Train_Accuracy', self.train_accuracy)
                if self.train_validate_split is not None:
                    self.validate_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                    self.var_validate_acc = tf.Variable([], dtype=tf.float32, trainable=False,
                                                        validate_shape=False, name='validate_accuracy_list')
                    self.update_validate_acc = tf.assign(self.var_validate_acc, self.list_validate_acc,
                                                         validate_shape=False)
                    if self.separate_writer is True:
                        self.validate_acc_summary = tf.summary.scalar('Validate_Accuracy', self.validate_accuracy)
                    else:
                        self.validate_acc_summary = tf.summary.scalar('Validation_Accuracy', self.validate_accuracy)
                if self.test_log is True:
                    self.test_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                    self.test_var_acc = tf.Variable([], dtype=tf.float32, trainable=False,
                                                    validate_shape=False, name='test_accuracy_list')
                    self.update_test_acc = tf.assign(self.test_var_acc, self.list_test_acc,
                                                     validate_shape=False)
                    if self.separate_writer is True:
                        self.test_acc_summary = tf.summary.scalar('Test_Accuracy', self.test_accuracy)
                    else:
                        self.test_acc_summary = tf.summary.scalar('Test_Accuracy', self.test_accuracy)

    def create_graph(self, num_features, num_classes):
        start = time.time()
        self.num_features = num_features
        self.num_classes = num_classes
        self.global_step = tf.Variable(0, name='last_successful_epoch', trainable=False, dtype=tf.int32)
        self.last_epoch = tf.assign(self.global_step, self.global_step + 1, name='assign_updated_epoch')
        # Step 1: Creating placeholders for inputs
        self.make_placeholders_for_inputs(num_features, num_classes)
        # Step 2: Creating initial parameters for the variables
        self.make_parameters(num_features, num_classes)
        # Step 3: Make predictions for the data
        self.make_predictions()
        # Step 4: Perform optimization operation
        self.make_optimization()
        # Step 5: Calculate accuracies
        self.make_accuracy()
        # Step 6: Initialize all the required variables
        with tf.device(self.device):
            self.init_var = tf.global_variables_initializer()
        end = time.time()
        print('Tensorflow graph created in %.4f seconds' %(end-start))
        return True

    def fit(self, data, labels, classes, test_data=None, test_labels=None, test_classes=None):
        pattern_cpu = re.compile('/cpu:[0-9]')
        pattern_gpu = re.compile('/gpu:[0-9]')
        if pattern_cpu.match(self.device):
            print('Using CPU: ', self.device)
            config = tf.ConfigProto(
                log_device_placement=True,
                allow_soft_placement=True,
                #allow_growth=True,
                #device_count={'CPU': 0}
            )
        if pattern_gpu.match(self.device):
            print('Using GPU: ', self.device)
            config = tf.ConfigProto(
                log_device_placement=True,
                allow_soft_placement=True,
                #allow_growth=True,
                #device_count={'GPU': 0}
            )
        if self.session_type == 'default':
            self.session = tf.Session(config=config)
        if self.session_type == 'interactive':
            self.session = tf.InteractiveSession(config=config)
        print('Session: ' + str(self.session))
        self.session.run(self.init_var)
        if self.tensorboard_logs is True:
            file_utils.mkdir_p(self.tensorboard_log_dir)
            self.merged_summary_op = tf.summary.merge_all()
            if self.restore is False:
                file_utils.delete_all_files_in_dir(self.tensorboard_log_dir)
            if self.separate_writer is False:
                self.summary_writer = tf.summary.FileWriter(self.tensorboard_log_dir, graph=self.session.graph)
            else:
                self.train_writer = tf.summary.FileWriter(self.tensorboard_log_dir + 'train',
                                                            graph=self.session.graph)
                if self.train_validate_split is not None:
                    self.validate_writer = tf.summary.FileWriter(self.tensorboard_log_dir + 'validate',
                                                              graph=self.session.graph)
                if self.test_log is True:
                    self.test_writer = tf.summary.FileWriter(self.tensorboard_log_dir + 'test',
                                                              graph=self.session.graph)
            if self.save_model is True:
                self.model = tf.train.Saver(max_to_keep=2)
        if self.train_validate_split is not None:
            train_data, validate_data, train_labels, validate_labels, train_classes, validate_classes = \
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
                self.optimize(train_data, train_labels, train_classes,
                              validate_data=validate_data, validate_labels=validate_labels,
                              validate_classes=validate_classes)
            else:
                self.optimize(train_data, train_labels, train_classes,
                              validate_data=validate_data, validate_labels=validate_labels,
                              validate_classes=validate_classes, test_data=test_data,
                              test_labels=test_labels, test_classes=test_classes)
        else:
            if self.test_log is False:
                self.optimize(data, labels, classes)
            else:
                self.optimize(data, labels, classes, test_data=test_data,
                              test_labels=test_labels, test_classes=test_classes)

    def optimize(self, train_data, train_labels, train_classes,
                 validate_data=None, validate_labels=None, validate_classes=None,
                 test_data=None, test_labels=None, test_classes=None):
        if self.transform is True:
            train_data = data_transform.transform(train_data, transform_method=self.transform_method)
            if self.train_validate_split is not None:
                validate_data = data_transform.transform(validate_data, transform_method=self.transform_method)
            if self.test_log is True:
                test_data = data_transform.transform(test_data, transform_method=self.transform_method)
        file_name = os.path.splitext(os.path.abspath(self.model_name))[0]
        num_files = len(sorted(glob.glob(os.path.abspath(file_name + '*.meta'))))
        if num_files > 0:
            checkpoint_file = os.path.abspath(sorted(glob.glob(file_name + '*.data-00000-of-00001'), reverse=True)[0])
            if os.path.exists(checkpoint_file):
                print('Restoring model from %s' % checkpoint_file)
                meta_file = os.path.abspath(sorted(glob.glob(file_name + '*.meta'), reverse=True)[0])
                print('Loading: %s' %meta_file)
                saver = tf.train.import_meta_graph(meta_file)
                print('Loading: %s' %os.path.abspath(checkpoint_file))
                cpk = tf.train.latest_checkpoint(os.path.dirname(meta_file))
                print('Checkpoint: ' + str(cpk))
                print('Tensors')
                print(print_tensors_in_checkpoint_file(file_name=cpk, all_tensors='', tensor_name=''))
                saver.restore(self.session, tf.train.latest_checkpoint(os.path.dirname(meta_file)))
                print('Last epoch to restore: ' + str(self.session.run(self.global_step)))
        if self.train_validate_split is not None:
            if self.test_log is False:
                self.run(train_data, train_labels, train_classes,
                              validate_data=validate_data, validate_labels=validate_labels,
                              validate_classes=validate_classes)
            else:
                self.run(train_data, train_labels, train_classes,
                         validate_data=validate_data, validate_labels=validate_labels,
                         validate_classes=validate_classes, test_data=test_data,
                         test_labels=test_labels, test_classes=test_classes)
        else:
            if self.test_log is False:
                self.run(train_data, train_labels, train_classes)
            else:
                self.run(train_data, train_labels, train_classes,
                         test_data=test_data, test_labels=test_labels, test_classes=test_classes)

    def run(self, train_data, train_labels, train_classes,
            validate_data=None, validate_labels=None, validate_classes=None,
            test_data=None, test_labels=None, test_classes=None):
        feed_dict_train = {self.x: train_data,
                           self.y_true: train_labels,
                           self.y_true_cls: train_classes}
        if self.train_validate_split is not None:
            feed_dict_validate = {self.x: validate_data,
                                  self.y_true: validate_labels,
                                  self.y_true_cls: validate_classes}
        if self.test_log is True:
            feed_dict_test = {self.x: test_data,
                              self.y_true: test_labels,
                              self.y_true_cls: test_classes}
        epoch = self.session.run(self.global_step)
        self.list_train_loss = self.session.run(self.var_train_loss)
        self.list_train_loss = self.list_train_loss.tolist()
        self.list_train_acc = self.session.run(self.var_train_acc)
        self.list_train_acc = self.list_train_acc.tolist()
        print('Length of train loss          : %d' %len(self.list_train_loss))
        print('Length of train accuracy      : %d' % len(self.list_train_acc))
        if self.train_validate_split is not None:
            self.list_validate_loss = self.session.run(self.var_validate_loss)
            self.list_validate_loss = self.list_validate_loss.tolist()
            self.list_validate_acc = self.session.run(self.var_validate_acc)
            self.list_validate_acc = self.list_validate_acc.tolist()
        print('Length of validate loss       : %d' % len(self.list_validate_loss))
        print('Length of validate accuracy   : %d' % len(self.list_validate_acc))
        if self.test_log is True:
            self.list_test_acc = self.session.run(self.test_var_acc)
            self.list_test_acc = self.list_test_acc.tolist()
        print('Length of test accuracy       : %d' % len(self.list_test_acc))
        print('Restoring training from epoch :', epoch)
        converged = False
        prev_cost = 0
        num_batches = int(train_data.shape[0] / self.batch_size)
        while (epoch != self.max_iterations) and converged is False:
            start = time.time()
            start_batch_index = 0
            for batch in range(num_batches):
                # print('Training on batch %d' %batch)
                end_batch_index = start_batch_index + self.batch_size
                if end_batch_index < train_data.shape[0]:
                    train_batch_data = train_data[start_batch_index:end_batch_index, :]
                    train_batch_labels = train_labels[start_batch_index:end_batch_index, :]
                    train_batch_classes = train_classes[start_batch_index:end_batch_index]
                else:
                    train_batch_data = train_data[start_batch_index:, :]
                    train_batch_labels = train_labels[start_batch_index:, :]
                    train_batch_classes = train_classes[start_batch_index:]
                feed_dict_train = {self.x: train_batch_data,
                                   self.y_true: train_batch_labels,
                                   self.y_true_cls: train_batch_classes}
                _, train_loss, train_acc \
                    = self.session.run([self.optimizer, self.train_loss, self.train_accuracy], feed_dict=feed_dict_train)
                train_loss_summary = self.session.run(self.train_loss_summary, feed_dict=feed_dict_train)
                train_acc_summary = self.session.run(self.train_acc_summary, feed_dict=feed_dict_train)
                start_batch_index += self.batch_size
            curr_epoch = self.session.run(self.last_epoch)
            validate_loss_summary = self.session.run(self.validate_loss_summary, feed_dict=feed_dict_validate)
            learning_rate_summary = self.session.run(self.learning_rate_summary)
            self.list_train_loss.append(train_loss)
            self.update_train_loss = tf.assign(self.var_train_loss, self.list_train_loss, validate_shape=False)
            self.update_train_loss.eval()
            self.list_train_acc.append(train_acc)
            self.update_train_acc = tf.assign(self.var_train_acc, self.list_train_acc, validate_shape=False)
            self.update_train_acc.eval()
            self.list_learning_rate.append(self.current_learning_rate)
            self.update_learning_rate = tf.assign(self.var_learning_rate, self.list_learning_rate, validate_shape=False)
            self.update_learning_rate.eval()
            # w_hist = self.session.run(self.w_hist, feed_dict=feed_dict_train)
            # self.summary_writer.add_summary(w_hist, epoch)
            # w_im = self.session.run(self.w_im, feed_dict=feed_dict_train)
            # self.summary_writer.add_summary(w_im, epoch)
            if self.train_validate_split is not None:
                validate_loss, validate_acc, validate_acc_summary = \
                self.session.run([self.validate_loss, self.validate_accuracy, self.validate_acc_summary],
                                 feed_dict=feed_dict_validate)
                self.list_validate_loss.append(validate_loss)
                self.update_validate_loss = tf.assign(self.var_validate_loss, self.list_validate_loss, validate_shape=False)
                self.update_validate_loss.eval()
                self.list_validate_acc.append(validate_acc)
                self.update_validate_acc = tf.assign(self.var_validate_acc, self.list_validate_acc, validate_shape=False)
                self.update_validate_acc.eval()
            if self.test_log is True:
                test_acc, test_acc_summary = \
                    self.session.run([self.test_accuracy, self.test_acc_summary], feed_dict=feed_dict_test)
                self.list_test_acc.append(test_acc)
                self.update_test_acc = tf.assign(self.test_var_acc, self.list_test_acc, validate_shape=False)
                self.update_test_acc.eval()
            if self.separate_writer is False:
                self.summary_writer.add_summary(train_loss_summary, epoch)
                self.summary_writer.add_summary(train_acc_summary, epoch)
                self.summary_writer.add_summary(validate_loss_summary, epoch)
                self.summary_writer.add_summary(validate_acc_summary, epoch)
                self.summary_writer.add_summary(test_acc_summary, epoch)
                self.summary_writer.add_summary(learning_rate_summary, epoch)
            else:
                self.train_writer.add_summary(train_loss_summary, epoch)
                self.train_writer.add_summary(train_acc_summary, epoch)
                if self.train_validate_split is not None:
                    self.validate_writer.add_summary(validate_loss_summary, epoch)
                    self.validate_writer.add_summary(validate_acc_summary, epoch)
                if self.test_log is True:
                    self.test_writer.add_summary(test_acc_summary, epoch)
            if epoch % self.display_step == 0:
                duration = time.time() - start
                if self.train_validate_split is not None and self.test_log is False:
                    print('>>> Epoch [%*d/%*d]'
                          %(int(len(str(self.max_iterations))), epoch,
                            int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | val_loss: %.4f | val_acc: %.4f | '
                          'Time: %.4f s' %(train_loss, train_acc, validate_loss, validate_acc, duration))
                elif self.train_validate_split is not None and self.test_log is True:
                    print('>>> Epoch [%*d/%*d]'
                          % (int(len(str(self.max_iterations))), epoch,
                             int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | val_loss: %.4f | val_acc: %.4f | '
                          'test_acc: %.4f | Time: %.4f s'
                          %(train_loss, train_acc, validate_loss, validate_acc, test_acc, duration))
                elif self.train_validate_split is None and self.test_log is True:
                    print('>>> Epoch [%*d/%*d]'
                          % (int(len(str(self.max_iterations))), epoch,
                             int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | test_acc: %.4f | Time: %.4f s'
                          %(train_loss, train_acc, test_acc, duration))
                else:
                    print('>>> Epoch [%*d/%*d]'
                          % (int(len(str(self.max_iterations))), epoch,
                             int(len(str(self.max_iterations))), self.max_iterations))
                    print('train_loss: %.4f | train_acc: %.4f | Time: %.4f s'
                          % (train_loss, train_acc, duration))
            if self.save_model is True:
                model_directory = os.path.dirname(self.model_name)
                file_utils.mkdir_p(model_directory)
                self.model.save(self.session, self.model_name, global_step=epoch)
            if epoch == 0:
                prev_cost = train_loss
            else:
                if math.fabs(train_loss-prev_cost) < self.tolerance:
                    converged = False
            epoch += 1

    def predict(self, data):
        if self.transform is True:
            data = data_transform.transform(data, transform_method=self.transform_method)
        feed_dict_data = {self.x: data}
        predictions = self.session.run(self.y_pred_cls, feed_dict=feed_dict_data)
        predictions = np.array(predictions)
        return predictions

    def load_model(self, model_name):
        self.model.restore(self.session, model_name)

    def close(self):
        self.session.close()

    def print_accuracy(self, test_data, test_labels, test_classes):
        predict_classes = self.predict(test_data)
        return accuracy_score(test_classes, predict_classes, normalize=True)

    def print_classification_results(self, test_data, test_labels, test_classes, test_class_names=[],
                                     normalize=True):
        if self.transform is True:
            test_data = data_transform.transform(test_data, transform_method=self.transform_method)
        feed_dict_test = {self.x: test_data,
                          self.y_true: test_labels,
                          self.y_true_cls: test_classes}
        cls_true = test_classes
        cls_pred = self.session.run(self.y_pred_cls, feed_dict=feed_dict_test)
        plot_tools.plot_confusion_matrix(cls_true, cls_pred, classes=test_class_names,
                              normalize=normalize, title='Confusion matrix for test dataset using multi-layer '
                                                         'perceptron')
        print('Detailed classification report')
        print(classification_report(y_true=cls_true, y_pred=cls_pred, target_names=test_class_names))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        if self.separate_writer is False:
            self.summary_writer.close()
        else:
            self.train_writer.close()
            if self.train_validate_split is not None:
                self.validate_writer.close()
            if self.test_log is True:
                self.test_writer.close()

    def __del__(self):
        self.session.close()
        if self.separate_writer is False:
            self.summary_writer.close()
        else:
            self.train_writer.close()
            if self.train_validate_split is not None:
                self.validate_writer.close()
            if self.test_log is True:
                self.test_writer.close()

