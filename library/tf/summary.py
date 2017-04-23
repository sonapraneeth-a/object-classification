import tensorflow as tf


class Summaries:

    def __init__(self,
                 train_loss=None,
                 validate_loss=None,
                 test_loss=None,
                 train_accuracy=None,
                 validate_accuracy=None,
                 test_accuracy=None,
                 verbose=False):
        self.train = True
        self.validate = True
        self.test = True
        self.train_writer = None
        self.validate_writer = None
        self.test_writer = None
        self.summary_writer = None
        self.train_loss = tf.summary.scalar('Train_error', train_loss)
        self.validate_loss = tf.summary.scalar('Validation_error', validate_loss)
        self.test_loss = tf.summary.scalar('Test_error', test_loss)
        self.train_acc = tf.summary.scalar('Train_accuracy', train_accuracy)
        self.validate_acc = tf.summary.scalar('Validation_accuracy', self.validate_accuracy)
        self.test_acc = tf.summary.scalar('Test_accuracy', self.test_accuracy)
        self.verbose = verbose

    def writeLoss(self):
        return True

    def writeAccuracy(self):
        return True