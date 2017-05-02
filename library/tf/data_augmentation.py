# Taken from tflearn library

import tensorflow as tf
import scipy.ndimage
import random
import numpy as np


def flip_up_down(images, labels, classes, random_seed=None):
    result_images = []
    result_labels = []
    result_classes = []
    np.random.seed(random_seed)
    for i in range(images.shape[0]):
        if random_seed is not None:
            if bool(random.getrandbits(1)):
                image = np.fliplr(images[i])
                result_images.append(image)
                result_labels.append(labels[i])
                result_classes.append(classes[i])
        else:
            image = np.fliplr(images[i])
            result_images.append(image)
            result_labels.append(labels[i])
            result_classes.append(classes[i])
    result_images = np.array(result_images)
    result_labels = np.array(result_labels)
    result_classes = np.array(result_classes)
    return result_images, result_labels, result_classes


def flip_left_right(images, labels, classes, random_seed=None):
    result_images = []
    result_labels = []
    result_classes = []
    np.random.seed(random_seed)
    for i in range(images.shape[0]):
        if random_seed is not None:
            if bool(random.getrandbits(1)):
                image = np.fliplr(images[i])
                result_images.append(image)
                result_labels.append(labels[i])
                result_classes.append(classes[i])
        else:
            image = np.fliplr(images[i])
            result_images.append(image)
            result_labels.append(labels[i])
            result_classes.append(classes[i])
    result_images = np.array(result_images)
    result_labels = np.array(result_labels)
    result_classes = np.array(result_classes)
    return result_images, result_labels, result_classes


def rotate90(images, labels, classes, k=1, random_seed=None):
    result_images = []
    result_labels = []
    result_classes = []
    np.random.seed(random_seed)
    for i in range(images.shape[0]):
        if random_seed is not None:
            if bool(random.getrandbits(1)):
                image = np.rot90(images[i], k)
                result_images.append(image)
                result_labels.append(labels[i])
                result_classes.append(classes[i])
        else:
            image = np.fliplr(images[i])
            result_images.append(image)
            result_labels.append(labels[i])
            result_classes.append(classes[i])
    result_images = np.array(result_images)
    result_labels = np.array(result_labels)
    result_classes = np.array(result_classes)
    return result_images, result_labels, result_classes


def random_rotate(images, labels, classes, max_angle=20, random_seed=None):
    result_images = []
    result_labels = []
    result_classes = []
    np.random.seed(random_seed)
    for i in range(images.shape[0]):
        if random_seed is not None:
            if bool(random.getrandbits(1)):
                angle = random.uniform(-max_angle, max_angle)
                image = scipy.ndimage.interpolation.rotate(images[i], angle,
                                                           reshape=False)
                result_images.append(image)
                result_labels.append(labels[i])
                result_classes.append(classes[i])
        else:
            angle = random.uniform(-max_angle, max_angle)
            image = scipy.ndimage.interpolation.rotate(images[i], angle,
                                                       reshape=False)
            result_images.append(image)
            result_labels.append(labels[i])
            result_classes.append(classes[i])
    result_images = np.array(result_images)
    result_labels = np.array(result_labels)
    result_classes = np.array(result_classes)
    return result_images, result_labels, result_classes


def transpose(images, k=1):
    result = tf.map_fn(lambda img: tf.image.transpose_image(img), images)
    with tf.Session() as sess:
        result = sess.run(result)
    return result