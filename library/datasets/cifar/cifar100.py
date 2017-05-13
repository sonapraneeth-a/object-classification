from library.utils import file_utils
import os, shutil
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from library.datasets.dataset import Dataset
from library.datasets.cifar.base import CIFARBase


class CIFAR100(CIFARBase):

    def __init__(self,
                 num_images=1.0,
                 one_hot_encode=False,
                 train_validate_split=None,
                 preprocess='',
                 augment=False,
                 num_test_images=1.0,
                 endian='little',
                 save_h5py='',
                 make_image=True,
                 image_mode='rgb',
                 verbose=False):
        super().__init__(num_images=num_images,
                         one_hot_encode=one_hot_encode,
                         train_validate_split=train_validate_split,
                         preprocess=preprocess,
                         augment=augment,
                         num_test_images=num_test_images,
                         endian=endian,
                         make_image=make_image,
                         image_mode=image_mode,
                         save_h5py=save_h5py,
                         verbose=verbose)
        self.num_fine_classes = 20
        self.num_coarse_classes = 100
        self.file_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.file_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
        self.fine_classes = \
            ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
             'household electrical devices', 'household furniture', 'insects', 'large carnivores',
             'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
             'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
             'trees', 'vehicles 1', 'vehicles 2']
        self.coarse_classes = \
            ['beaver', 'dolphin', 'otter', 'seal', 'whale',
             'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
             'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
             'bottles', 'bowls', 'cans', 'cups', 'plates',
             'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
             'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
             'bed', 'chair', 'couch', 'table', 'wardrobe',
             'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
             'bear', 'leopard', 'lion', 'tiger', 'wolf',
             'bridge', 'castle', 'house', 'road', 'skyscraper',
             'cloud', 'forest', 'mountain', 'plain', 'sea',
             'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
             'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
             'crab', 'lobster', 'snail', 'spider', 'worm',
             'baby', 'boy', 'girl', 'man', 'woman',
             'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
             'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
             'maple', 'oak', 'palm', 'pine', 'willow',
             'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
             'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

    def download_and_extract_data(self, data_directory):
        print('Downloading and extracting CIFAR 100 file')
        ## Step 1: Make the directories './datasets/cifar100/' if they do not exist
        if not os.path.exists(data_directory):
            if self.verbose is True:
                print('Creating the directory \'%s\'' % data_directory)
            file_utils.mkdir_p(data_directory)
        else:
            if self.verbose is True:
                print('Directory \'%s\' already exists' % data_directory)
        ## Step 2: Check if './datasets/cifar100/cifar-10-python.tar.gz' exists
        tar_file = data_directory + 'cifar-100.tar.gz'
        make_tar = False
        if not os.path.exists(tar_file):
            make_tar = True
        elif os.path.exists(tar_file) and not file_utils.verify_md5(tar_file, self.file_md5):
            if self.verbose is True:
                print('Removing the wrong file \'%s\'' % tar_file)
            os.remove(tar_file)
            make_tar = True
        else:
            if self.verbose is True:
                print('CIFAR 10 tarfile exists and MD5 sum is verified')
        ## Step 3: Download CIFAR 100 dataset from 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        if make_tar is True:
            result = file_utils.download(self.file_url, tar_file, verbose=self.verbose)
            if result is False:
                if self.verbose is True:
                    raise FileNotFoundError('Download of CIFAR 100 dataset failed')
                return False
            result = file_utils.verify_md5(tar_file, self.file_md5, verbose=self.verbose)
            if result is False:
                if self.verbose is True:
                    raise FileNotFoundError('Downloaded CIFAR 100 dataset failed md5sum check')
                return False
        ## Step 4: Extract the dataset
        make_extract = False
        batches_directory = data_directory + 'cifar-10-batches'
        if not os.path.exists(batches_directory):
            make_extract = True
        else:
            num_files = sum(os.path.isfile(os.path.join(batches_directory, f))
                            for f in os.listdir(batches_directory))
            if num_files != 8:
                shutil.rmtree(batches_directory)
                make_extract = True
            else:
                if self.verbose is True:
                    print('Directory %s already exists' %batches_directory)
        if make_extract is True:
            print('Extracting file %s to %s' %(tar_file,batches_directory))
            result = file_utils.extract(tar_file)
            shutil.move('./cifar-10-batches-py', batches_directory)
            if result is False:
                if self.verbose is True:
                    print('Extraction of CIFAR 100 dataset failed')
                return False
            else:
                if self.verbose is True:
                    print('Extraction of CIFAR 100 dataset success')
        return True

    def dict_read(self, dict_file):
        fine_labels = dict_file[b'fine_labels']
        coarse_labels = dict_file[b'coarse_labels']
        data = dict_file[b'data']
        batch_label = dict_file[b'batch_label']
        filenames = dict_file[b'filenames']
        if self.verbose is True:
            print(dict_file.keys())
            print(fine_labels[:5])
            print(coarse_labels[:5])
            print(batch_label[:5])
            print(filenames[:5])
            print(data[0].shape)
        return data, fine_labels, coarse_labels, batch_label, filenames

    def load_train_data(self, split=False, train_validate_split=0.8, data_directory='./tmp/cifar100/'):
        print('Loading CIFAR 100 Training Dataset')
        basic_dir_path = data_directory + 'cifar-10-batches/'
        data_batch_path = 'data_batch_'
        data_files = []
        data_dict = []
        for i in range(1, 6):
            data_files.append(str(basic_dir_path + data_batch_path + str(i)))
        for file in data_files:
            print('Unpickling data file: %s' % file)
            data_dict.append(file_utils.unpickle(file))
        data_labels = []
        data_images = []
        for i in range(len(data_dict) - 1):
            print('Reading unpicked data file: %s' %data_files[i])
            data, labels, _, _, _ = self.dict_read(data_dict[i])
            data_labels.extend(labels)
            data_images.extend(data)
        self.train.data['train_images'] = np.array(data_images)
        self.train.class_labels['train_labels'] = np.array(data_labels)
        del data_labels
        del data_images
        return True

    def load_test_data(self, data_directory='/tmp/cifar100/'):
        print('Loading CIFAR 100 Test Dataset')
        basic_dir_path = data_directory + 'cifar-10-batches/'
        test_batch_path = 'test_batch'
        test_files = [str(basic_dir_path + test_batch_path)]
        print('Unpickling test file: %s' % test_files[0])
        test_dict = [file_utils.unpickle(test_files[0])]
        test_labels = []
        test_images = []
        print('Reading unpicked test file: %s' % test_files[0])
        test_labels.extend(self.dict_read(test_dict[-1])[1])
        test_images.extend(self.dict_read(test_dict[-1])[0])
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        self.data['test_images'] = np.array(test_images)
        self.data['test_labels'] = np.array(test_labels)
        del test_labels
        del test_images
        return True

    def load_data(self, train=True, test=True, data_directory='/tmp/cifar100/'):
        print('Loading CIFAR 100 dataset')
        self.download_and_extract_data(data_directory)
        if train is True:
            self.load_train_data(data_directory=data_directory)
        if test is True:
            self.load_test_data(data_directory=data_directory)
        return True
