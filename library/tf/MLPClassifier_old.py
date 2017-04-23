import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from library.utils import file_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import os, glob, time
from os.path import basename
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from library.preprocessing import ZCA

# Resources
# https://www.youtube.com/watch?v=3VEXX73tnw4


class TFMLPClassifier:

    def __init__(self, logs=True, log_dir='./logs/', learning_rate=0.01, activation_fn='softmax', restore=True,
                 num_iterations=100, device='cpu', session_type='default', descent_method='adam',
                 init_weights='random', display_step=10, reg_const=0.01, regularize=False,
                 learning_rate_type='constant', model_name='./model/mlp_classifier_model.ckpt',
                 save_model=False, transform=True, test_log=True, transform_method='StandardScaler',
                 tolerance=1e-7, train_validate_split=None, separate_writer=False, batch_size=100,
                 nodes_in_layers=[5, 5], activation_req=False, verbose=False):
        # Docs
        self.tensorboard_logs = logs
        self.verbose = verbose
        self.tensorboard_log_dir = log_dir
        self.merged_summary_op = None
        self.summary_writer = None
        self.train_writer = None
        self.validate_writer = None
        self.test_writer = None
        self.model = None
        self.model_name = model_name
        self.save_model = save_model
        self.train_loss_summary = None
        self.train_acc_summary = None
        self.validate_acc_summary = None
        self.test_acc_summary = None
        self.w_hist = None
        self.w_im = None
        self.b_hist = None
        self.restore = restore
        self.separate_writer = separate_writer
        #
        self.session = None
        self.device = device
        self.session_type = session_type
        # Parameters
        self.learning_rate = learning_rate
        self.max_iterations = num_iterations
        self.display_step = display_step
        self.tolerance = tolerance
        self.descent_method = descent_method
        self.init_weights = init_weights
        self.reg_const = reg_const
        self.regularize = regularize
        self.activation = activation_fn
        self.learning_rate_type = learning_rate_type
        self.batch_size = batch_size
        self.hidden_layers = nodes_in_layers
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
        self.train_accuracy = None
        self.validate_accuracy = None
        self.test_log = test_log
        self.test_accuracy = None
        self.activation_req = activation_req
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.output_layer = None
        self.loss = None
        self.correct_prediction = None
        self.cross_entropy = None
        #
        self.train_validate_split = train_validate_split

    def print_parameters(self):
        print('Linear Classifier')

    def make_placeholders_for_inputs(self, num_features, num_classes):
        with tf.device('/cpu:0'):
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
            weight = None
            weight_key = 'weight_' + str(layer_no)
            if self.init_weights == 'zeros':
                weight = tf.Variable(tf.zeros([prev_layer_weights, self.hidden_layers[layer_no]]),
                                     name='W_'+str(layer_no)+'_zeros')
            elif self.init_weights == 'random':
                weight = tf.Variable(tf.random_normal([prev_layer_weights, self.hidden_layers[layer_no]]),
                                           name='W_'+str(layer_no)+'_random_normal')
            else:
                weight = tf.Variable(tf.random_normal([prev_layer_weights, self.hidden_layers[layer_no]]),
                                           name='W_'+str(layer_no)+'_random_normal')
            self.weights[weight_key] = weight
            prev_layer_weights = self.hidden_layers[layer_no]

    def make_bias(self, num_features, num_classes, number_of_layers=1):
        for layer_no in range(number_of_layers):
            bias = None
            bias_key = 'bias_' + str(layer_no)
            bias = tf.Variable(tf.random_normal([self.hidden_layers[layer_no]]), name='b_'+str(layer_no))
            self.biases[bias_key] = bias

    def make_layers(self, number_of_layers):
        prev_layer = self.x
        for layer_no in range(number_of_layers):
            layer = None
            weight_key = 'weight_' + str(layer_no)
            bias_key = 'bias_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(tf.add(tf.matmul(prev_layer, self.weights[weight_key]), self.biases[bias_key]),
                                  name='Layer_'+str(layer_no)+'_sigmoid')
            elif self.activation == 'softmax':
                layer = tf.nn.softmax(tf.add(tf.matmul(prev_layer, self.weights[weight_key]), self.biases[bias_key]),
                                      name='Layer_' + str(layer_no)+'_softmax')
            elif self.activation == 'relu':
                layer = tf.nn.relu(tf.add(tf.matmul(prev_layer, self.weights[weight_key]), self.biases[bias_key]),
                                      name='Layer_' + str(layer_no)+'_relu')
            else:
                layer = tf.nn.relu(tf.add(tf.matmul(prev_layer, self.weights[weight_key]), self.biases[bias_key]),
                                   name='Layer_' + str(layer_no)+'_relu')
            self.layers[layer_key] = layer
            prev_layer = self.layers[layer_key]

    def make_output_layer(self):
        layer_key = layer_key = 'layer_' + str(len(self.hidden_layers)-1)
        print(len(self.hidden_layers))
        print(layer_key)
        print(self.weights.keys())
        print(self.biases.keys())
        print('output')
        output = tf.Variable( tf.random_normal([self.hidden_layers[-1], self.num_classes]))
        print('bias')
        bias_output = tf.Variable(tf.random_normal([self.num_classes]))
        print('layer')
        self.output_layer = tf.add(tf.matmul(self.layers[layer_key], output), bias_output, name='out_layer')

    def make_parameters(self, num_features, num_classes):
        with tf.device('/cpu:0'):
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
        with tf.device('/cpu:0'):
            with tf.name_scope('Predictions'):
                if self.activation_req is True:
                    if self.activation == 'softmax':
                        self.y_pred = tf.nn.softmax(self.output_layer)
                    elif self.activation == 'relu':
                        self.y_pred = tf.nn.relu(self.output_layer)
                    elif self.activation == 'sigmoid':
                        self.y_pred = tf.nn.sigmoid(self.output_layer)
                else:
                    self.y_pred = self.output_layer
                self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

    def make_optimization(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Cross_Entropy'):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer,
                                                                             labels=self.y_true)
            with tf.name_scope('Loss_Function'):
                if self.regularize is True:
                    ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                    ridge_loss = tf.reduce_mean(tf.square(self.weights))
                    self.loss = tf.add(tf.reduce_mean(self.cross_entropy), tf.multiply(ridge_param, ridge_loss))
                else:
                    self.loss = tf.reduce_mean(self.cross_entropy)
                self.train_loss_summary = tf.summary.scalar('Training_Error', self.loss)
            with tf.name_scope('Optimizer'):
                if self.learning_rate_type == 'exponential':
                    learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                               self.display_step, 0.96, staircase=True)
                else:
                    learning_rate = self.learning_rate
                if self.descent_method == 'gradient':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
                        .minimize(self.loss)
                elif self.descent_method == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                        .minimize(self.loss)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
                        .minimize(self.loss)
            self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

    def make_accuracy(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Train_Accuracy'):
                self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                if self.separate_writer is True:
                    self.train_acc_summary = tf.summary.scalar('Train_Accuracy', self.train_accuracy)
                else:
                    self.train_acc_summary = tf.summary.scalar('Train_Accuracy', self.train_accuracy)
            if self.train_validate_split is not None:
                with tf.name_scope('Validate_Accuracy'):
                    self.validate_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                    if self.separate_writer is True:
                        self.validate_acc_summary = tf.summary.scalar('Validate_Accuracy', self.validate_accuracy)
                    else:
                        self.validate_acc_summary = tf.summary.scalar('Validation_Accuracy', self.validate_accuracy)
            if self.test_log is True:
                with tf.name_scope('Test_Accuracy'):
                    self.test_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                    if self.separate_writer is True:
                        self.test_acc_summary = tf.summary.scalar('Test_Accuracy', self.test_accuracy)
                    else:
                        self.test_acc_summary = tf.summary.scalar('Test_Accuracy', self.test_accuracy)

    def create_graph(self, num_features, num_classes):
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
        with tf.device('/cpu:0'):
            self.init_var = tf.global_variables_initializer()
        if self.verbose is True:
            print('X                  : ' + str(self.x))
            print('Y_true             : ' + str(self.y_true))
            print('Y_true_cls         : ' + str(self.y_true_cls))
            print('W                  : ' + str(self.weights))
            for i in range(len(self.hidden_layers)):
                weight_key = 'weight_' + str(i)
                print('  weight_%d        : %s' % (i,str(self.weights[weight_key])))
            print('b                  : ' + str(self.biases))
            for i in range(len(self.hidden_layers)):
                bias_key = 'bias_' + str(i)
                print('  bias_%d          : %s' % (i,str(self.biases[bias_key])))
            print('Layers             : ' + str(self.layers))
            for i in range(len(self.hidden_layers)):
                layer_key = 'layer_' + str(i)
                print('  layer_%d         : %s' % (i,str(self.layers[layer_key])))
            print('Output layer       : ' + str(self.output_layer))
            print('Y_pred             : ' + str(self.y_pred))
            print('Y_pred_cls         : ' + str(self.y_pred_cls))
            print('cross_entropy      : ' + str(self.cross_entropy))
            print('train_loss               : ' + str(self.loss))
            print('optimizer          : ' + str(self.optimizer))
            print('correct_prediction : ' + str(self.correct_prediction))
            print('Train Accuracy     : ' + str(self.train_accuracy))
            print('Validate Accuracy  : ' + str(self.validate_accuracy))
            print('Test Accuracy      : ' + str(self.test_accuracy))
        return True

    def fit(self, data, labels, classes, test_data=None, test_labels=None, test_classes=None):
        if self.device == 'cpu':
            print('Using CPU')
            config = tf.ConfigProto(
                log_device_placement=True,
                allow_soft_placement=True,
                #allow_growth=True,
                #device_count={'CPU': 0}
            )
        else:
            print('Using GPU')
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
                self.model = tf.train.Saver(max_to_keep=5)
        if self.train_validate_split is not None:
            train_data, validate_data, train_labels, validate_labels, train_classes, validate_classes = \
                train_test_split(data, labels, classes, train_size=self.train_validate_split)
            if self.verbose is True:
                print('Data shape: ' + str(data.shape))
                print('Labels shape: ' + str(labels.shape))
                print('Classes shape: ' + str(classes.shape))
                print('Train Data shape: ' + str(train_data.shape))
                print('Train Labels shape: ' + str(train_labels.shape))
                print('Train Classes shape: ' + str(train_classes.shape))
                print('Validate Data shape: ' + str(validate_data.shape))
                print('Validate Labels shape: ' + str(validate_labels.shape))
                print('Validate Classes shape: ' + str(validate_classes.shape))
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
        if self.train_validate_split is not None:
            feed_dict_validate = {self.x: validate_data,
                                  self.y_true: validate_labels,
                                  self.y_true_cls: validate_classes}
        if self.test_log is True:
            feed_dict_test = {self.x: test_data,
                              self.y_true: test_labels,
                              self.y_true_cls: test_classes}
        epoch = self.session.run(self.global_step)
        print('Last successful epoch: ' + str(epoch))
        converged = False
        prev_cost = 0
        start = time.time()
        end_batch_index = 0
        num_batches = int(train_data.shape[0] / self.batch_size)
        while (epoch != self.max_iterations) and converged is False:
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
                _, cost, train_acc, curr_epoch = self.session.run([self.optimizer, self.loss, self.train_accuracy,
                                                            self.last_epoch], feed_dict=feed_dict_train)
                train_loss_summary = self.session.run(self.train_loss_summary, feed_dict=feed_dict_train)
                train_acc_summary = self.session.run(self.train_acc_summary, feed_dict=feed_dict_train)
                start_batch_index += self.batch_size
            if self.train_validate_split is not None:
                validate_acc, validate_summary = \
                self.session.run([self.validate_accuracy, self.validate_acc_summary],
                                 feed_dict=feed_dict_validate)
            if self.test_log is True:
                test_acc, test_summary = \
                    self.session.run([self.test_accuracy, self.test_acc_summary],
                                     feed_dict=feed_dict_test)
            if self.separate_writer is False:
                self.summary_writer.add_summary(train_loss_summary, epoch)
                self.summary_writer.add_summary(train_acc_summary, epoch)
                self.summary_writer.add_summary(validate_summary, epoch)
                self.summary_writer.add_summary(test_summary, epoch)
            else:
                self.train_writer.add_summary(train_loss_summary, epoch)
                self.train_writer.add_summary(train_acc_summary, epoch)
                if self.train_validate_split is not None:
                    self.validate_writer.add_summary(validate_summary, epoch)
                if self.test_log is True:
                    self.test_writer.add_summary(test_summary, epoch)
            if epoch % self.display_step == 0:
                duration = time.time() - start
                if self.train_validate_split is not None and self.test_log is False:
                    print('>>> Epoch [%*d/%*d] | Error: %.4f | Train Acc.: %.4f | Validate Acc.: %.4f | '
                          'Duration: %.4f seconds'
                          %(int(len(str(self.max_iterations))), epoch, int(len(str(self.max_iterations))),
                                self.max_iterations, cost, train_acc, validate_acc, duration))
                elif self.train_validate_split is not None and self.test_log is True:
                    print('>>> Epoch [%*d/%*d] | Error: %.4f | Train Acc.: %.4f | Validate Acc.: %.4f | '
                          'Test Acc.: %.4f | Duration: %.4f seconds'
                          %(int(len(str(self.max_iterations))), epoch, int(len(str(self.max_iterations))),
                                self.max_iterations, cost, train_acc, validate_acc, test_acc, duration))
                elif self.train_validate_split is None and self.test_log is True:
                    print('>>> Epoch [%*d/%*d] | Error: %.4f | Train Acc.: %.4f | '
                          'Test Acc.: %.4f | Duration: %.4f seconds'
                          %(int(len(str(self.max_iterations))), epoch, int(len(str(self.max_iterations))),
                                self.max_iterations, cost, train_acc, test_acc, duration))
                else:
                    print('>>> Epoch [%*d/%*d] | Error: %.4f | Train Acc.: %.4f | Duration of run: %.4f seconds'
                          % (int(len(str(self.max_iterations))), epoch, int(len(str(self.max_iterations))),
                             self.max_iterations, cost, train_acc))
            start = time.time()
            if self.save_model is True:
                model_directory = os.path.dirname(self.model_name)
                file_utils.mkdir_p(model_directory)
                self.model.save(self.session, self.model_name, global_step=epoch)
            if epoch == 0:
                prev_cost = cost
            else:
                if math.fabs(cost-prev_cost) < self.tolerance:
                    converged = False
            epoch += 1
            # print('Current success step: ' + str(self.session.run(self.global_step)))

    def fit_and_test(self, data, labels, classes, test_data, test_labels, test_classes):
        self.fit(data, labels, classes)

    def predict(self, data):
        if self.transform is True:
            if self.transform_method == 'StandardScaler':
                ss = StandardScaler()
                data = ss.fit_transform(data)
            if self.transform_method == 'MinMaxScaler':
                ss = MinMaxScaler()
                data = ss.fit_transform(data)
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

    def print_classification_results(self, test_data, test_labels, test_classes):
        if self.transform is True:
            if self.transform_method == 'StandardScaler':
                ss = StandardScaler()
                test_data = ss.fit_transform(test_data)
            if self.transform_method == 'MinMaxScaler':
                ss = MinMaxScaler()
                test_data = ss.fit_transform(test_data)
        feed_dict_test = {self.x: test_data,
                          self.y_true: test_labels,
                          self.y_true_cls: test_classes}
        cls_true = test_classes
        cls_pred = self.session.run(self.y_pred_cls, feed_dict=feed_dict_test)
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        print('Confusion matrix')
        print(cm)
        print('Detailed classification report')
        print(classification_report(y_true=cls_true, y_pred=cls_pred))

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
