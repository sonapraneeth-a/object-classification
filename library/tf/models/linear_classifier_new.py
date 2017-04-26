import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from scipy.misc import toimage
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from library.tf.parameters import Weights, Bias
from library.tf.layers import mlp_layer, optimize_algo
from library.tf.summary import Summaries
from library.plot_tools import plot
from library.utils import file_utils
import os, glob, time, math
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from library.tf.base import BaseClassifier


class TFLinearClassifier(BaseClassifier):

    def __init__(self, device='/cpu:0',
                 verbose=False,
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
                 save_model=False,
                 model_name='./model/cnn_classifier_model.ckpt',
                 restore=True,
                 descent_method='gradient',
                 config=None
                 ):
        BaseClassifier.__init__(self,
                                verbose=verbose,
                                device=device,
                                session_type=session_type,
                                num_iter=num_iter,
                                err_tolerance=err_tolerance,
                                train_validate_split=train_validate_split,
                                display_step=display_step,
                                learn_step=learn_step,
                                learn_rate_type=learn_rate_type,
                                learn_rate=learn_rate,
                                reg_const=reg_const,
                                logs=logs, log_dir=log_dir,
                                test_log=True, save_model=True, model_name=model_name,
                                restore=False, descent_method=descent_method, batch_size=batch_size)
        self.verbose = verbose
        # Classifier Parameter configuration
        self.config = config
        self.num_classes = None
        self.num_features = None
        self.graph = None

    def print_parameters(self):
        return True

    def make_placeholders_for_inputs(self, graph):
        with graph.as_default():
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

    def make_parameters(self, graph):
        with graph.as_default():
            with tf.device(self.device):
                with tf.name_scope('Parameters'):
                    with tf.name_scope('Weights'):
                        self.model_params['weight'] = \
                            Weights.define((self.num_features, self.num_classes),
                                           weight_name=self.config['weight']['name'],
                                           weight_type=self.config['weight']['type'])
                    with tf.name_scope('Bias'):
                        self.model_params['bias'] = \
                            Bias.define((self.num_classes,),
                                        bias_name=self.config['bias']['name'],
                                        bias_type=self.config['bias']['type'])

    def create_graph(self, num_classes, num_features):
        start = time.time()
        self.num_features = num_features
        self.num_classes = num_classes
        self.graph = tf.Graph()
        # Step 1: Creating placeholders for inputs
        self.make_placeholders_for_inputs(self.graph)
        # Step 2: Creating initial parameters for the variables
        self.make_parameters(self.graph)
        # Step 3: Make predictions for the data
        final_layer = mlp_layer(self.predict_params['input'], self.model_params['weight'],
                                  self.model_params['bias'],
                                  activation_type=self.config['activation_fn'])
        self.make_predictions(self.graph, final_layer, self.num_classes)
        # Step 4: Perform optimization operation
        self.make_optimization(self.graph, final_layer)
        # Step 5: Calculate accuracies
        self.make_accuracy(self.graph)
        self.start_session(self.graph)
        end = time.time()
        print('Graph completed')

    def plot_weights(self, classes=[], fig_size=(16,12), fontsize=10):
        weights = self.session.run(self.model_params['weight'])
        if len(classes) == 0:
            classes = []
            for image_no in range(10):
                classes.append('Class ' + str(image_no))
        images = []
        for image_no in range(10):
            images.append(weights[:, image_no].reshape([3, 32, 32]).transpose([1, 2, 0]))
        images = np.array(images)
        fig, axes = plt.subplots(1, 10)
        if fig_size is not None:
            fig.set_figheight(fig_size[0])
            fig.set_figwidth(fig_size[1])
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        type = 'rgb'
        for image_no, ax in enumerate(axes.flat):
            # Plot image.
            image = images[image_no, :]
            if type == 'rgb':
                ax.imshow(toimage(image), cmap='binary')
            elif type == 'grey':
                ax.imshow(toimage(image), cmap=matplotlib.cm.Greys_r)
            else:
                ax.imshow(image, cmap='binary')
            ax.set_xlabel(classes[image_no], weight='bold', size=fontsize)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        # plt.title('Learnt weights from linear classifier')
        # plt.tight_layout()
        plt.show()
        return True