import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from library.tf import Layers
from library.plot_tools import plot_tools
from library.preprocessing import data_transform
import numpy as np
from library.utils import file_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import os, glob, time, re
from os.path import basename
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from library.preprocessing import ZCA

# Resources
# https://www.youtube.com/watch?v=3VEXX73tnw4


class TFLinearClassifier:

    def __init__(self, logs=True, log_dir='./logs/', learning_rate=0.01, activation_fn='softmax', restore=True,
                 num_iterations=100, device='', session_type='default', descent_method='gradient',
                 init_weights='random', display_step=10, reg_const=0.01, regularize=False, init_bias='ones',
                 learning_rate_type='constant', model_name='./model/linear_classifier_model.ckpt',
                 save_model=False, transform=True, test_log=True, transform_method='StandardScaler',
                 tolerance=1e-7, train_validate_split=None, separate_writer=False, verbose=False):
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
        self.init_weights = init_weights
        self.init_bias = init_bias
        self.reg_const = reg_const
        self.regularize = regularize
        self.activation_fn = activation_fn
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
        self.weights = None
        self.biases = None
        self.logits = None
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
        print('Weights initializer        : ', self.init_weights)
        print('Bias initializer           : ', self.init_bias)
        print('>> Inputs for Tensorflow Graph')
        print('X                          : ', self.x)
        print('Y_true                     : ', self.y_true)
        print('Y_true_cls                 : ', self.y_true_cls)
        print('Device to use              : ', self.device)
        print('>> Output parameters for Tensorflow Graph')
        print('Restore model              : ', self.restore)
        print('W                          : ', self.weights)
        print('b                          : ', self.biases)
        print('logits                     : ', self.logits)
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

    def make_parameters(self, num_features, num_classes):
        with tf.device(self.device):
            with tf.name_scope('Parameters'):
                with tf.name_scope('Weights'):
                    if self.init_weights == 'zeros':
                        self.weights = tf.Variable(tf.zeros([num_features, num_classes]), name='W_zeros')
                    elif self.init_weights == 'ones':
                        self.weights = tf.Variable(tf.ones([num_features, num_classes]), name='W_ones')
                    elif self.init_weights == 'random_normal':
                        self.weights = tf.Variable(tf.random_normal([num_features, num_classes]), name='W_random_normal')
                    else:
                        self.weights = tf.Variable(tf.random_normal([num_features, num_classes]), name='W_random_normal')
                with tf.name_scope('Bias'):
                    if self.init_bias == 'zeros':
                        self.biases = tf.Variable(tf.zeros([num_classes]), name='b_zeros')
                    elif self.init_bias == 'ones':
                        self.biases = tf.Variable(tf.ones([num_classes]), name='b_ones')
                    elif self.init_bias == 'random_normal':
                        self.biases = tf.Variable(tf.random_normal([num_classes]), name='b_random_normal')
                    else:
                        self.biases = tf.Variable(tf.random_normal([num_classes]), name='b_random_normal')
                self.w_hist = tf.summary.histogram('Weights_Histogram', self.weights)
                self.w_im = tf.summary.image('Weights_Image', self.weights)
                self.b_hist = tf.summary.histogram('Bias', self.biases)

    def make_predictions(self):
        with tf.device(self.device):
            with tf.name_scope('Predictions'):
                self.logits = tf.matmul(self.x, self.weights) + self.biases
                if self.activation_fn == 'softmax':
                    self.y_pred = tf.nn.softmax(self.logits)
                elif self.activation_fn == 'relu':
                    self.y_pred = tf.nn.relu(self.logits)
                elif self.activation_fn == 'sigmoid':
                    self.y_pred = tf.nn.sigmoid(self.logits)
                self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

    def make_optimization(self):
        with tf.device(self.device):
            with tf.name_scope('Cross_Entropy'):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_true)
            with tf.name_scope('Loss_Function'):
                if self.regularize is True:
                    ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                    ridge_loss = tf.reduce_mean(tf.square(self.weights))
                    self.train_loss = tf.add(tf.reduce_mean(self.cross_entropy), tf.multiply(ridge_param, ridge_loss))
                    if self.train_validate_split is not None:
                        self.validate_loss = tf.add(tf.reduce_mean(self.cross_entropy), tf.multiply(ridge_param, ridge_loss))
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
                if self.descent_method == 'gradient':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.current_learning_rate)\
                        .minimize(self.train_loss)
                elif self.descent_method == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning_rate)\
                        .minimize(self.train_loss)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.current_learning_rate)\
                        .minimize(self.train_loss)
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
                 test_data = None, test_labels = None, test_classes = None):
        if self.transform is True:
            if self.transform_method == 'StandardScaler':
                ss = StandardScaler()
                train_data = ss.fit_transform(train_data)
                if self.train_validate_split is not None:
                    validate_data = ss.fit_transform(validate_data)
                if self.test_log is True:
                    test_data = ss.fit_transform(test_data)
            if self.transform_method == 'MinMaxScaler':
                ss = MinMaxScaler()
                train_data = ss.fit_transform(train_data)
                if self.train_validate_split is not None:
                    validate_data = ss.fit_transform(validate_data)
                if self.test_log is True:
                    test_data = ss.fit_transform(test_data)
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
        while (epoch != self.max_iterations) and converged is False:
            start = time.time()
            _, train_loss, train_acc, curr_epoch \
                = self.session.run([self.optimizer, self.train_loss, self.train_accuracy,
                                    self.last_epoch], feed_dict=feed_dict_train)
            train_loss_summary = self.session.run(self.train_loss_summary, feed_dict=feed_dict_train)
            validate_loss_summary = self.session.run(self.validate_loss_summary, feed_dict=feed_dict_validate)
            train_acc_summary = self.session.run(self.train_acc_summary, feed_dict=feed_dict_train)
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
                              normalize=normalize, title='Confusion matrix')
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

