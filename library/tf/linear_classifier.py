import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from library.tf.parameters import Weights, Bias
from library.tf.layers import activation_layer, optimize_algo
from library.plot_tools import plot_tools
import numpy as np
from library.utils import file_utils
import os, glob, time, re, math
from os.path import basename
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from library.preprocessing import ZCA


class LinearClassifier:

    def __init__(self, device='/cpu:0',
                 verbose=True,
                 session_type='default',
                 num_iter=100,
                 err_tolerance=1e-7,
                 train_validate_split=None,
                 display_step=10,
                 learn_step=10,
                 transform=True,
                 transform_method='StandardScaler',
                 learn_rate_type='constant',
                 learn_rate=0.01,
                 reg_const=0.0,
                 logs=True,
                 log_dir='./logs/',
                 save_model=False,
                 model_name='./model/linear_classifier_model.ckpt',
                 restore=True,
                 config={'weight': {'type': 'random_normal', 'name': 'Weight'},
                         'bias': {'type': 'random_normal', 'name': 'Bias'},
                         'activation_fn': 'softmax',
                         'descent_method': 'gradient'},
                 ):
        self.verbose = verbose
        # Classifier Parameter configuration
        self.config = config
        self.model_params = dict()                 # weights, bias
        self.params = dict()                       # any other params
        self.predict_params = dict()               # input, class output, one hot output
        self.out_params = dict()                   # loss and accuracy
        self.out_params['list_train_loss'] = []
        self.out_params['list_train_acc'] = []
        self.out_params['list_val_loss'] = []
        self.out_params['list_val_acc'] = []
        self.out_params['list_learn_rate'] = []
        self.summary_params = {}
        # Learning parameters
        self.learn_rate_type = learn_rate_type
        self.learn_rate = learn_rate
        self.reg_const = reg_const
        # Logging Parameters
        self.logging = logs
        self.logging_dir = log_dir
        # Model Parameters
        self.save_model = save_model
        self.model = None
        self.model_name = model_name
        self.restore = restore
        # Data transformation method
        self.transform = transform
        self.transform_method = transform_method
        # Running configuration
        self.device = device
        self.session_tye = session_type
        self.session = None
        self.max_iterations = num_iter
        self.err_tolerance = err_tolerance
        self.train_validate_split = train_validate_split
        if self.train_validate_split is None:
            self.train_validate_split = 0.8
        # External parameters
        self.display_step = display_step
        self.learn_step = learn_step

    @staticmethod
    def print_parameters():
        print('Parameters for simple linear classifier')

    def make_placeholders_for_inputs(self, num_features, num_classes):
        with tf.device(self.device):
            with tf.name_scope('Inputs'):
                with tf.name_scope('Data'):
                    self.predict_params['input'] = \
                        tf.placeholder(tf.float32,
                                       [None, num_features], name='X_input')
                with tf.name_scope('Train_Labels'):
                    self.predict_params['output_one_hot_labels'] = \
                        tf.placeholder(tf.float32, [None, num_classes], name='y_label')
                    self.predict_params['output_class'] = \
                        tf.placeholder(tf.int64, [None], name='y_class')

    def make_parameters(self):
        with tf.device(self.device):
            with tf.name_scope('Parameters'):
                with tf.name_scope('Weights'):
                    self.model_params['weight'] = \
                        Weights.define(self.config['weight']['shape'],
                                       weight_name=self.config['weight']['name'],
                                       weight_type=self.config['weight']['type'])
                with tf.name_scope('Bias'):
                    self.model_params['bias'] = \
                        Bias.define(self.config['bias']['shape'],
                                    bias_name=self.config['bias']['name'],
                                    bias_type=self.config['bias']['type'])
                w_hist = tf.summary.histogram('Weights_Histogram', self.model_params['weight'])
                w_im = tf.summary.image('Weights_Image', self.model_params['weight'])
                b_hist = tf.summary.histogram('Bias', self.model_params['bias'])

    def make_predictions(self):
        with tf.device(self.device):
            with tf.name_scope('Predictions'):
                self.params['logits'] = tf.matmul(self.predict_params['input'],
                                                  self.model_params['weight']) + \
                                        self.model_params['bias']
                self.predict_params['one_hot_output'] = \
                    activation_layer(self.params['logits'],
                                     activation_type=self.config['activation_fn'],
                                     activation_name='')
                self.predict_params['class_output'] = \
                    tf.argmax(self.params['one_hot_output'], dimension=1)

    def make_optimization(self):
        with tf.device(self.device):
            with tf.name_scope('Cross_Entropy'):
                self.params['cross_entropy'] = \
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.params['logits'],
                                                            labels=self.predict_params['one_hot_output'])
            with tf.name_scope('Loss_Function'):
                ridge_param = tf.cast(tf.constant(self.reg_const), dtype=tf.float32)
                ridge_loss = tf.reduce_mean(tf.square(self.model_params['weight']))
                self.summary_params['train_loss'] = \
                    tf.add(tf.reduce_mean(self.params['cross_entropy']),
                           tf.multiply(ridge_param, ridge_loss))
                self.summary_params['summary_train_loss'] = \
                    tf.summary.scalar('Training_Error', self.summary_params['train_loss'])
                self.summary_params['val_loss'] = \
                    tf.add(tf.reduce_mean(self.params['cross_entropy']),
                           tf.multiply(ridge_param, ridge_loss))
                self.summary_params['summary_val_loss'] = \
                    tf.summary.scalar('Validation_Error', self.summary_params['val_loss'])
                self.out_params['var_train_loss'] = \
                    tf.Variable([], dtype=tf.float32, trainable=False,
                                validate_shape=False, name='train_loss_list')
                self.out_params['update_train_loss'] = \
                    tf.assign(self.out_params['var_train_loss'],
                              self.out_params['list_train_loss'], validate_shape=False)
                self.out_params['var_val_loss'] = \
                    tf.Variable([], dtype=tf.float32, trainable=False,
                                validate_shape=False, name='validate_loss_list')
                self.out_params['update_val_loss'] = \
                    tf.assign(self.out_params['var_val_loss'],
                              self.out_params['list_val_loss'], validate_shape=False)
            with tf.name_scope('Optimizer'):
                self.out_params['var_learn_rate'] = \
                    tf.Variable([], dtype=tf.float32, trainable=False,
                                validate_shape=False,
                                name='learning_rate_progress')
                self.out_params['update_learn_rate'] = \
                    tf.assign(self.out_params['var_learn_rate'], self.out_params['list_learn_rate'],
                              validate_shape=False)
                if self.learning_rate_type == 'exponential':
                    self.current_learning_rate = \
                        tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.display_step, 0.96,
                                                   staircase=True)
                else:
                    self.current_learning_rate = self.learning_rate
                self.learning_rate_summary = tf.summary.scalar('Learning_rate', self.current_learning_rate)
                optimizer = optimize_algo(self.current_learning_rate, descent_method=self.config['descent_method'])
                optimizer.minimize(self.train_loss)
            self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

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
        with tf.device(self.device):
            self.init_var = tf.global_variables_initializer()
        end = time.time()
        print('Tensorflow graph created in %.4f seconds' % (end - start))
        return True

    def load_model(self, model_name):
        self.model.restore(self.session, model_name)

    def close(self):
        self.session.close()
    
    def fit(self):
        return True

    def optimize(self):
        return True
    
    def run(self):
        return True

    def predict(self, data):
        feed_dict_data = {self.predict_params['input']: data}
        predictions = self.session.run(self.predict_params['output_class'], feed_dict=feed_dict_data)
        predictions = np.array(predictions)
        return predictions
    
    def print_classification_results(self, test_data,
                                     test_labels,
                                     test_classes,
                                     test_class_names=[],
                                     normalize=True):
        feed_dict_test = {self.predict_params['input']: test_data,
                          self.predict_params['output_one_hot_labels']: test_labels,
                          self.predict_params['output_class']: test_classes}
        if len(test_class_names) == 0:
            unique_values = list(set(test_classes))
            num_classes = len(unique_values)
            test_class_names = []
            for classes in range(num_classes):
                test_class_names.append('Class '+str(classes))
        cls_true = test_classes
        cls_pred = self.session.run(self.predict_params['output_class'], feed_dict=feed_dict_test)
        plot_tools.plot_confusion_matrix(cls_true, cls_pred, classes=test_class_names,
                                         normalize=normalize, title='Confusion matrix')
        print('Detailed classification report')
        print(classification_report(y_true=cls_true, y_pred=cls_pred, target_names=test_class_names))

    def print_accuracy(self, test_data, test_classes):
        predict_classes = self.predict(test_data)
        return accuracy_score(test_classes, predict_classes, normalize=True)
    
    def score(self):
        return True
    
    def plot_loss(self):
        return True
    
    def plot_accuracy(self):
        return True

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