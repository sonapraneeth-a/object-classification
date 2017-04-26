import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from library.tf.parameters import Weights, Bias
from library.tf.layers import mlp_layer, optimize_algo
from library.tf.summary import Summaries
from library.plot_tools import plot
from library.utils import file_utils
import os, glob, time, math
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class MLPClassifier:
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
                 batch_size=128,
                 reg_const=0.0,
                 logs=True,
                 log_dir='./logs/',
                 test_log=False,
                 save_model=True,
                 save_checkpoint=True,
                 checkpoint_filename='mlp_classifier_model.ckpt',
                 model_name='mlp_classifier_model.pb',
                 descent_method='gradient',
                 restore=False,
                 config={'layer_1': {'weight': {'name': 'Weight_1', 'type': 'random_normal'},
                                     'bias': {'name': 'Bias_1', 'type': 'random_normal'},
                                     'activation_fn': 'sigmoid',
                                     'num_nodes': 10,
                                     'layer_name': 'Layer_1'
                                    },
                         'output_layer': {'weight': {'name': 'Output_Weight', 'type': 'random_normal'},
                                          'bias': {'name': 'Output_Bias', 'type': 'random_normal'},
                                          'activation_fn': 'sigmoid',
                                          'layer_name': 'Output_Layer'
                                         }
                         },
                 ):
        self.verbose = verbose
        # Classifier Parameter configuration
        self.config = config
        self.descent_method = descent_method
        self.num_classes = None
        self.num_features = None
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
        self.reg_const = reg_const
        # Logging Parameters
        self.logging = logs
        self.logging_dir = log_dir
        self.summary_writer = None
        self.test_log = test_log
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
        print('Parameters for MLP classifier')
        print('>> Input Parameters')
        print('Input                  : %s ' % str(self.predict_params['input']))
        print('True one hot labels    : %s ' % str(self.predict_params['true_one_hot']))
        print('True class             : %s ' % str(self.predict_params['true_class']))
        print('Predict one hot labels : %s ' % str(self.predict_params['predict_one_hot']))
        print('Predict class          : %s ' % str(self.predict_params['predict_class']))
        for layer_no in range(1, self.num_layers+1):
            weight_key = 'weight_' + str(layer_no)
            bias_key = 'bias_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            print('>> Layer %d' % layer_no)
            print('\tWeights : %s'
                  % str(self.model_params['weights'][weight_key]))
            print('\tBias    : %s'
                  % str(self.model_params['bias'][bias_key]))
            print('\tLayer   : %s'
                  % str(self.model_params['layers'][layer_key]))
        print('>> Outer Layer')
        print('\tWeights      : %s'
              % str(self.model_params['weights']['output_layer']))
        print('\tBias         : %s'
              % str(self.model_params['bias']['output_layer']))
        print('\tOutput Layer : %s'
              % str(self.model_params['layers']['output_layer']))
        print('Output Logits  : %s' % str(self.params['logits']))
        print('Entropy        : %s' % str(self.params['cross_entropy']))
        print('Predictions    : %s ' % str(self.params['predictions']))
        print('Optimizer      : %s ' % str(self.params['optimizer']))
        print('>> Model params')
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
                                       [None, self.num_features], name='X_input')
                with tf.name_scope('Train_Labels'):
                    self.predict_params['true_one_hot'] = \
                        tf.placeholder(tf.float32, [None, self.num_classes], name='y_true_one_hot_label')
                    self.predict_params['true_class'] = \
                        tf.placeholder(tf.int64, [None], name='y_true_class')

    def make_weights(self, number_of_layers=1):
        in_units = self.num_features
        for layer_no in range(1, number_of_layers+1):
            weight_key = 'weight_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            if 'type' in self.config[layer_key]['weight']:
                weight_type = self.config[layer_key]['weight']['type']
            else:
                weight_type = 'random_normal'
            if 'name' in self.config[layer_key]['weight']:
                weight_name = self.config[layer_key]['weight']['name']
            else:
                weight_name = 'Weight_' + str(layer_no)
            out_units = self.config[layer_key]['num_nodes']
            shape = (in_units, out_units)
            weight = Weights.define(shape, weight_type=weight_type, weight_name=weight_name)
            self.model_params['weights'][weight_key] = weight
            in_units = out_units

    def make_bias(self, number_of_layers=1):
        for layer_no in range(1, number_of_layers+1):
            bias_key = 'bias_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            if 'type' in self.config[layer_key]['bias']:
                bias_type = self.config[layer_key]['bias']['type']
            else:
                bias_type = 'random_normal'
            if 'name' in self.config[layer_key]['bias']:
                bias_name = self.config[layer_key]['bias']['name']
            else:
                bias_name = 'Bias_' + str(layer_no)
            out_units = self.config[layer_key]['num_nodes']
            shape = (out_units,)
            bias = Bias.define(shape, bias_type=bias_type, bias_name=bias_name)
            self.model_params['bias'][bias_key] = bias

    def make_layers(self, number_of_layers=1):
        prev_layer = self.predict_params['input']
        for layer_no in range(1, number_of_layers+1):
            weight_key = 'weight_' + str(layer_no)
            bias_key = 'bias_' + str(layer_no)
            layer_key = 'layer_' + str(layer_no)
            if 'name' in self.config[layer_key]:
                layer_name = self.config[layer_key]['name']
            else:
                layer_name = 'Layer_' + str(layer_no)
            if 'activation_fn' in self.config[layer_key]:
                activation_fn = self.config[layer_key]['activation_fn']
            else:
                activation_fn = 'sigmoid'
            weight = self.model_params['weights'][weight_key]
            bias = self.model_params['bias'][bias_key]
            layer = mlp_layer(prev_layer, weight, bias, layer_name=layer_name,
                              activation_type=activation_fn)
            self.model_params['layers'][layer_key] = layer
            prev_layer = self.model_params['layers'][layer_key]

    def make_output_layer(self):
        prev_layer_key = 'layer_' + str(self.num_layers)
        layer_key = 'output_layer'
        prev_layer = self.model_params['layers'][prev_layer_key]
        if 'name' in self.config[layer_key]:
            layer_name = self.config[layer_key]['name']
        else:
            layer_name = 'Output_Layer'
        if 'activation_fn' in self.config[layer_key]:
            activation_fn = self.config[layer_key]['activation_fn']
        else:
            activation_fn = 'sigmoid'
        if prev_layer_key in self.config:
            in_units = self.config[prev_layer_key]['num_nodes']
        else:
            in_units = self.num_features
        weight_shape = (in_units, self.num_classes)
        bias_shape = (self.num_classes,)
        if 'type' in self.config[layer_key]['weight']:
            weight_type = self.config[layer_key]['weight']['type']
        else:
            weight_type = 'random_normal'
        if 'name' in self.config[layer_key]['weight']:
            weight_name = self.config[layer_key]['weight']['name']
        else:
            weight_name = 'Weight_output'
        if 'type' in self.config[layer_key]['bias']:
            bias_type = self.config[layer_key]['bias']['type']
        else:
            bias_type = 'random_normal'
        if 'name' in self.config[layer_key]['bias']:
            bias_name = self.config[layer_key]['bias']['name']
        else:
            bias_name = 'Bias_output'
        weight = Weights.define(weight_shape, weight_type=weight_type, weight_name=weight_name)
        bias = Bias.define(bias_shape, bias_type=bias_type, bias_name=bias_name)
        self.model_params['weights']['output_layer'] = weight
        self.model_params['bias']['output_layer'] = bias
        self.model_params['layers']['output_layer'] = \
            mlp_layer(prev_layer, weight, bias, activation_type=activation_fn, layer_name=layer_name)

    def make_parameters(self, number_of_layers=1):
        with tf.device(self.device):
            with tf.name_scope('Parameters'):
                with tf.name_scope('Weights'):
                    self.make_weights(number_of_layers=number_of_layers)
                with tf.name_scope('Bias'):
                    self.make_bias(number_of_layers=number_of_layers)
            with tf.name_scope('Hidden_Layers'):
                self.make_layers(number_of_layers=number_of_layers)
            with tf.name_scope('Output_Layer'):
                self.make_output_layer()

    def make_predictions(self):
        with tf.device(self.device):
            with tf.name_scope('Predictions'):
                self.params['logits'] = tf.nn.softmax(self.model_params['layers']['output_layer'])
                self.predict_params['predict_class'] = \
                    tf.argmax(self.params['logits'], dimension=1)
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
            with tf.name_scope('Loss_Function'):
                ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                ridge_loss = \
                    tf.multiply(ridge_param,
                                tf.reduce_mean(tf.square(self.model_params['weights']['weight_1'])))
                for layer_no in range(2, self.num_layers):
                    weight_key = 'weight_' + str(layer_no)
                    ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                    ridge_loss = \
                        tf.add(ridge_loss,
                               tf.multiply(ridge_param,
                                           tf.reduce_mean(tf.square(self.model_params['weights'][weight_key]))))
                total_loss = tf.add(tf.reduce_mean(self.params['cross_entropy']), ridge_loss)
                # total_loss = tf.reduce_mean(self.params['cross_entropy'])
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

    def create_graph(self, num_features, num_classes):
        start = time.time()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = len(self.config.keys())-1
        self.global_step = tf.Variable(0, name='last_successful_epoch', trainable=False, dtype=tf.int32)
        self.last_epoch = tf.assign(self.global_step, self.global_step + 1, name='assign_updated_epoch')
        # Step 1: Creating placeholders for inputs
        self.make_placeholders_for_inputs()
        # Step 2: Creating initial parameters for the variables
        self.make_parameters(number_of_layers=self.num_layers)
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
            # allow_growth=True,
            # device_count={'GPU': 0}
        )
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
            if self.save_model is True:
                self.model = tf.train.Saver(max_to_keep=2)
        # Step 9: Restore model
        if self.restore is True:
            self.restore_model()
        epoch = self.session.run(self.global_step)
        print('Model has been trained for %d iterations' % epoch)
        end = time.time()
        print('Tensorflow graph created in %.4f seconds' % (end - start))
        return True

    def fit(self, data, labels, classes, test_data=None, test_labels=None, test_classes=None):
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
        num_batches = int(train_data.shape[0] / self.batch_size)
        while (epoch != self.max_iterations) and converged is False:
            start = time.time()
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
            if self.save_model is True:
                model_directory = os.path.dirname(self.model_name)
                file_utils.mkdir_p(model_directory)
                self.model.save(self.session, self.model_name, global_step=epoch)
            if epoch == 0:
                prev_cost = train_loss
            else:
                if math.fabs(train_loss - prev_cost) < self.err_tolerance:
                    converged = False
            epoch += 1
        end = time.time()
        print('Fit completed in %.4f seconds' % (end - start))
        if self.save_model is True:
            print('Saving the graph to %s' % (self.logging_dir+'/'+self.model_name))
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
        absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_folder + '/' + self.model_name.split('/')[-1]
        output_node_names = 'Predictions/predict_class'
        clear_devices = True
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        graph = self.session.graph
        input_graph_def = graph.as_graph_def()
        saver.restore(self.session, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            self.session,
            input_graph_def,
            output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))
        return True

    def load_checkpoint(self, checkpoint_filename):
        self.model.restore(self.session, checkpoint_filename)

    def close(self):
        self.session.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.summary_writer.close()
        self.session.close()

    def __del__(self):
        self.summary_writer.close()
        self.session.close()
