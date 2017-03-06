##############################################################################
#
# Functions for downloading the CIFAR-10, CIFAR-100 data-set from the internet
# and loading it into memory.
#
# Implemented and tested in Python 3.6
#
# Usage:
#
# Format:
#
##############################################################################

from utils import file_utils
import os
import shutil
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)

## CIFAR 10 Variables from website
CIFAR10_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']

## CIFAR 10 Variables from Kaggle

## CIFAR 100 Variables
CIFAR100_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def make_cifar10_data(verbose=True):
    print('Making CIFAR 10 dataset')
    output_directory = './data/cifar10/website/'
    output_data_file = output_directory + 'cifar10_data.csv'
    output_test_file = output_directory + 'cifar10_test.csv'
    ## Step 1: Make the directories './data/cifar10' if they do not exist
    if not os.path.exists(output_directory):
        if verbose is True:
            print('Creating the directory \'%s\'' % output_directory)
        os.makedirs(output_directory)
    else:
        if verbose is True:
            print('Directory \'%s\' already exists' % output_directory)
    ## Step 2:
    # 2.1 Check if './data/cifar10/website/cifar10_data.csv' exists
    # 2.2 Check if './data/cifar10/website/cifar10_test.csv' exists
    # 2.3 Check if './data/cifar10/website/cifar-10.tar.gz' exists
    make_data = False
    make_test = False
    if not os.path.exists(output_data_file):
        make_data = True
    if not os.path.exists(output_test_file):
        make_test = True
    if not (make_data or make_test):
        print('Both files exist')
        return True
    ## Step 2: Check if './data/cifar10/cifar-10.tar.gz' exists
    tar_file = output_directory + 'cifar-10.tar.gz'
    make_tar = False
    if not os.path.exists(tar_file):
        make_tar = True
    elif os.path.exists(tar_file) and not file_utils.verify_md5(tar_file, CIFAR10_MD5):
        if verbose is True:
            print('Removing the wrong file \'%s\'' % tar_file)
        os.remove(tar_file)
        make_tar = True
    else:
        print('Tar file for CIFAR 10 dataset exists and MD5 sum is verified')
    ## Step 4: Download CIFAR 10 dataset from 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if make_tar is True:
        result = file_utils.download(CIFAR10_URL, tar_file, verbose=verbose)
        if result is False:
            if verbose is True:
                print('Download of CIFAR 10 dataset failed')
            return False
        result = file_utils.verify_md5(tar_file, CIFAR10_MD5, verbose=verbose)
        if result is False:
            if verbose is True:
                print('Downloaded CIFAR 10 dataset failed md5sum check')
            return False
    ## Step 5: Extract the dataset from cifar-10.tar.gz
    make_extract = False
    if not os.path.exists('./data/cifar10/website/cifar10-batches'):
        make_extract = True
    else:
        path = output_directory + 'cifar10-batches'
        num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
        if num_files != 8:
            shutil.rmtree(path)
            make_extract = True
        else:
            print('Directory already exists')
    if make_extract is True:
        result = file_utils.extract(tar_file)
        shutil.move('./cifar-10-batches-py', './data/cifar10/website/cifar10-batches')
        if result is False:
            if verbose is True:
                print('Extraction of CIFAR 10 dataset failed')
            return False
        else:
            if verbose is True:
                print('Extraction of CIFAR 10 dataset success')
    ## Step 6: Make the './data/cifar10_data.csv' and './data/cifar10_test.csv'
    basic_dir_path = output_directory + 'cifar10-batches/'
    data_batch_path = 'data_batch_'
    test_batch_path = 'test_batch'
    data_files = []
    data_dict = []
    for i in range(1, 6):
        data_files.append(str(basic_dir_path + data_batch_path + str(i)))
    data_files.append(str(basic_dir_path + test_batch_path))
    for file in data_files:
        data_dict.append(file_utils.unpickle(file))
    data_labels = []
    test_labels = []
    data_images = []
    test_images = []
    for i in range(len(data_dict)-1):
        data, labels = dict_read(data_dict[i])
        data_labels.extend(labels)
        data_images.extend(data)
    test_labels.extend(dict_read(data_dict[-1])[1])
    test_images.extend(dict_read(data_dict[-1])[0])
    data_labels = np.array(data_labels)
    data_images = np.array(data_images)
    test_labels = np.array(test_labels)
    test_images = np.array(test_images)
    print('Length of train labels: ' + str(data_labels.shape))
    print('Length of train images: ' + str(data_images.shape))
    print('Length of test labels: ' + str(test_labels.shape))
    print('Length of test images: ' + str(test_images.shape))
    data_column_list = ['ID']
    test_column_list = ['ID']
    for i in range(1, test_images.shape[1]+1):
        label = 'pixel' + str(i)
        data_column_list.append(label)
        test_column_list.append(label)
    data_column_list.append('Class')
    test_column_list.append('Class')
    data_column_names = np.array(data_column_list)
    test_column_names = np.array(test_column_list)
    data_index = np.arange(1, data_images.shape[0]+1).reshape(data_images.shape[0], 1)
    test_index = np.arange(1, test_images.shape[0]+1).reshape(test_images.shape[0], 1)
    data = np.column_stack((data_index, data_images))
    data = np.column_stack((data, data_labels))
    test = np.column_stack((test_index, test_images))
    test = np.column_stack((test, test_labels))
    print('Data Shape: ' + str(data.shape))
    print('Test Shape: ' + str(test.shape))
    print('Data Index Shape: ' + str(data_index.shape))
    print('Test Index Shape: ' + str(test_index.shape))
    print('Data Column Header Shape: ' + str(data_column_names.shape))
    print('Test Column Header Shape: ' + str(test_column_names.shape))
    print(data[:5, :5])
    print(test[:5, :5])
    print('Making Pandas dataframes: data')
    data_frame = pd.DataFrame(data=data,
                              columns=data_column_names)
    print('Making Pandas dataframes: test')
    test_frame = pd.DataFrame(data=test,
                              columns=test_column_names)
    print('Writing data to \'./data/cifar10/cifar10_data.csv\'')
    data_frame.to_csv('./data/cifar10/cifar10_data.csv', index=False)
    print('Writing data to \'./data/cifar10/cifar10_test.csv\'')
    test_frame.to_csv('./data/cifar10/cifar10_test.csv', index=False)
    print('Completed making CIFAR 10 dataset')
    return True


def load_cifar10(data=True, test=True, verbose=False):
    directory = './data/cifar10/website/'
    data_csv = directory + 'cifar10_data.csv'
    test_csv = directory + 'cifar10_test.csv'
    make_data = False
    make_test = False
    if not os.path.exists(data_csv):
        make_data = True
    if not os.path.exists(test_csv):
        make_test = True
    if not (make_data or make_test):
        if data is True:
            print('Reading CIFAR 10 data')
            data_df = pd.read_csv(data_csv, index_col='ID')
            data_labels = data_df['Class']
            data_df = data_df.drop('Class', 1)
            data_images = data_df.as_matrix()
            print(data_images.shape)
        if test is True:
            print('Reading CIFAR 10 Test Data')
            test_df = pd.read_csv(test_csv, index_col='ID')
            test_labels = test_df['Class']
            test_df = test_df.drop('Class', 1)
            test_images = test_df.as_matrix()
            print(test_images.shape)
        if data is True and test is True:
            return data_images, data_labels, test_images, test_labels
        if data is True and test is False:
            return data_images, data_labels
        if data is True and test is True:
            return test_images, test_labels
        else:
            return None
    else:
        make_cifar10_data(verbose=False)


def load_cifar100(data=True, test=True, verbose=False):
    directory = './data/cifar100/website/'
    data_csv = directory + 'cifar100_data.csv'
    test_csv = directory + 'cifar100_test.csv'
    make_data = False
    make_test = False
    if not os.path.exists(data_csv):
        make_data = True
    if not os.path.exists(test_csv):
        make_test = True
    if not (make_data or make_test):
        if data is True:
            print('Reading CIFAR 100 data')
            data_df = pd.read_csv(data_csv, index_col='ID')
            data_labels = data_df['Class']
            data_df = data_df.drop('Class', 1)
            data_images = data_df.as_matrix()
            print(data_images.shape)
        if test is True:
            print('Reading CIFAR 100 Test Data')
            test_df = pd.read_csv(test_csv, index_col='ID')
            test_labels = test_df['Class']
            test_df = test_df.drop('Class', 1)
            test_images = test_df.as_matrix()
            print(test_images.shape)
        if data is True and test is True:
            return data_images, data_labels, test_images, test_labels
        if data is True and test is False:
            return data_images, data_labels
        if data is True and test is True:
            return test_images, test_labels
        else:
            return None
    else:
        raise IOError('Files donot exist')


def dict_read(dict_file):
    print(dict_file.keys())
    labels = dict_file[b'labels']
    data = dict_file[b'data']
    batch_label = dict_file[b'batch_label']
    filenames = dict_file[b'filenames']
    print(labels[:5])
    print(batch_label[:5])
    print(filenames[:5])
    print(data[0].shape)
    return data, labels


def convert_images(raw, type='rgb'):
    if type=='float':
        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0
    elif type=='rgb':
        raw_float = np.array(raw, dtype=float)
    else:
        print('Unknown format')
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, 32, 32, 3], order='F')
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 1, 3])
    images = images[0, :]
    return images


def plot_sample_cifar10():
    data_images, data_labels = load_cifar10(test=False)
    image_nos = range(20)
    example_images = data_images[image_nos, :]
    example_image_matrix = []
    k = 0
    for i in range(len(image_nos)):
        example_image_matrix.append(convert_images(example_images[k, :]))
        k += 1
    example_image_matrix = np.array(example_image_matrix)
    f, axarr = plt.subplots(ncols=2, sharex=True, sharey=True)
    f.subplots_adjust(wspace=0,hspace=0.05)
    k = 0
    num_cols = 10
    num_rows = int(len(image_nos)/num_cols)
    print(num_rows)
    print(num_cols)
    print(example_image_matrix.size)
    for i in range(num_rows):
        for j in range(num_cols):
            axarr[i, j].axis('off')
            axarr[i, j].imshow(toimage(example_image_matrix[k, :]))
            k += 1
    plt.show()