from library.utils import file_utils
import os, shutil, time
import numpy as np
from library.preprocessing.data_transform import transform
from library.datasets.cifar.base import CIFARBase
import h5py


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
        """
        
        :param num_images: 
        :param one_hot_encode: 
        :param train_validate_split: 
        :param preprocess: 
        :param augment: 
        :param num_test_images: 
        :param endian: 
        :param save_h5py: 
        :param make_image: 
        :param image_mode: 
        :param verbose: 
        """
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
        self.coarse_classes = \
            ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
             'household electrical devices', 'household furniture', 'insects', 'large carnivores',
             'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
             'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
             'trees', 'vehicles 1', 'vehicles 2']
        self.fine_classes = \
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
        """
        
        :param data_directory: 
        :return: 
        """
        print('Downloading and extracting CIFAR 100 file')
        ## Step 1: Make the directories './datasets/cifar100/' if they do not exist
        if not os.path.exists(data_directory):
            if self.verbose is True:
                print('Creating the directory \'%s\'' % data_directory)
            file_utils.mkdir_p(data_directory)
        else:
            if self.verbose is True:
                print('Directory \'%s\' already exists' % data_directory)
        ## Step 2: Check if './datasets/cifar100/cifar-100-python.tar.gz' exists
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
                print('CIFAR 100 tarfile exists and MD5 sum is verified')
        ## Step 3: Download CIFAR 100 dataset from 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
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
        batches_directory = data_directory + 'cifar-100-batches'
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
            shutil.move('./cifar-100-python', batches_directory)
            if result is False:
                if self.verbose is True:
                    print('Extraction of CIFAR 100 dataset failed')
                return False
            else:
                if self.verbose is True:
                    print('Extraction of CIFAR 100 dataset success')
        return True

    def dict_read(self, dict_file):
        """
        
        :param dict_file: 
        :return: 
        """
        print(dict_file.keys())
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

    def load_train_data(self, data_directory='/tmp/cifar100/'):
        """
        
        :param data_directory: 
        :return: 
        """
        print('Loading CIFAR 100 Train Dataset')
        basic_dir_path = data_directory + 'cifar-100-batches/'
        data_batch_path = 'train'
        data_files = [basic_dir_path + data_batch_path]
        data_dict = [file_utils.unpickle(data_files[0])]
        print('Reading unpicked data file: %s' % data_files[0])
        data, fine_labels, coarse_labels, _, _ = self.dict_read(data_dict[0])
        print(np.max(fine_labels))
        print(np.max(coarse_labels))
        data_fine_labels = fine_labels
        data_coarse_labels = coarse_labels
        data_images = np.array(data)
        data_fine_labels = np.array(data_fine_labels)
        data_coarse_labels = np.array(data_coarse_labels)
        print('Success')
        preprocessed_images = transform(data_images, transform_method=self.preprocess)
        if self.make_image is True:
            images = []
            for fig_num in range(preprocessed_images.shape[0]):
                fig = preprocessed_images[fig_num, :]
                img = self.convert_images(fig, type=self.image_mode)
                images.append(img)
            images = np.array(images)
        if self.train_validate_split is None:
            self.train.data = np.array(preprocessed_images[:self.num_images, :])
            if self.make_image is True:
                self.train.images = np.array(images[:self.num_images, :])
            self.train.fine_labels = np.array(data_fine_labels[:self.num_images])
            self.train.coarse_labels = np.array(data_coarse_labels[:self.num_images])
            self.train.fine_class_names = np.array(list(map(lambda x: self.fine_classes[x], self.train.fine_labels)))
            print(self.fine_classes[:15])
            print(self.train.fine_labels[:15])
            print(self.train.fine_class_names[:15])
            self.train.coarse_class_names = np.array(list(map(lambda x: self.coarse_classes[x], self.train.coarse_labels)))

        else:
            print('Requested to use only %d images' %self.num_images)
            self.train.data = np.array(preprocessed_images[:self.num_train_images, :])
            if self.make_image is True:
                self.train.images = np.array(images[:self.num_train_images, :])
            self.train.fine_labels = np.array(data_fine_labels[:self.num_train_images])
            self.train.coarse_labels = np.array(data_coarse_labels[:self.num_train_images])

            self.train.fine_class_names = np.array(list(map(lambda x: self.fine_classes[x], self.train.fine_labels)))
            self.train.coarse_class_names = np.array(list(map(lambda x: self.coarse_classes[x], self.train.coarse_labels)))
            self.validate.data = \
                np.array(preprocessed_images[self.num_train_images:self.num_train_images+self.num_validate_images, :])
            if self.make_image is True:
                self.validate.images = np.array(images[self.num_train_images:self.num_train_images+self.num_validate_images, :])
            self.validate.fine_labels = \
                np.array(data_fine_labels[self.num_train_images:self.num_train_images+self.num_validate_images])
            self.validate.coarse_labels = \
                np.array(data_coarse_labels[self.num_train_images:self.num_train_images + self.num_validate_images])
            self.validate.fine_class_names = np.array(list(map(lambda x: self.fine_classes[x],
                                                               self.validate.fine_labels)))
            self.validate.coarse_class_names = np.array(list(map(lambda x: self.coarse_classes[x],
                                                                 self.validate.coarse_labels)))
        if self.one_hot_encode is True:
            self.convert_one_hot_encoding(self.train.fine_labels, data_type='train', class_type='fine')
            self.convert_one_hot_encoding(self.train.coarse_labels, data_type='train', class_type='coarse')
            if self.train_validate_split is not None:
                self.convert_one_hot_encoding(self.validate.fine_labels, data_type='validate', class_type='fine')
                self.convert_one_hot_encoding(self.validate.coarse_labels, data_type='validate', class_type='coarse')

        if self.save_h5py != '':
            h5f = h5py.File(self.save_h5py, 'a')
            h5f.create_dataset('train_dataset', data=self.train.data, compression="gzip", compression_opts=9)
            print('Written CIFAR 100 train dataset to file: %s' % self.save_h5py)
            h5f.close()
        del data_coarse_labels
        del data_fine_labels
        del data_images
        del preprocessed_images
        if self.make_image is True:
            del images
        print()
        return True

    def load_test_data(self, data_directory='/tmp/cifar100/'):
        """
        
        :param data_directory: 
        :return: 
        """
        print('Loading CIFAR 100 Test Dataset')
        basic_dir_path = data_directory + 'cifar-100-batches/'
        test_batch_path = 'test'
        test_files = [str(basic_dir_path + test_batch_path)]
        print('Unpickling test file: %s' % test_files[0])
        test_dict = [file_utils.unpickle(test_files[0])]
        data, fine_labels, coarse_labels, _, _ = self.dict_read(test_dict[0])
        test_fine_labels = fine_labels
        test_coarse_labels = coarse_labels
        print('Reading unpicked test file: %s' % test_files[0])
        test_images = data
        test_images = np.array(test_images)
        preprocessed_images = transform(test_images, transform_method=self.preprocess)
        if self.make_image is True:
            images = []
            for fig_num in range(preprocessed_images.shape[0]):
                fig = preprocessed_images[fig_num, :]
                img = self.convert_images(fig, type=self.image_mode)
                images.append(img)
            images = np.array(images)
        test_fine_labels = np.array(test_fine_labels)
        test_coarse_labels = np.array(test_coarse_labels)
        self.test.data = np.array(preprocessed_images[:self.num_test_images])
        if self.make_image is True:
            self.test.images = np.array(images[:self.num_test_images, :])
        self.test.fine_labels = np.array(test_fine_labels[:self.num_test_images])
        self.test.coarse_labels = np.array(test_coarse_labels[:self.num_test_images])
        self.test.fine_class_names = np.array(list(map(lambda x: self.fine_classes[x], self.test.fine_labels)))
        self.test.coarse_class_names = np.array(list(map(lambda x: self.coarse_classes[x], self.test.coarse_labels)))

        if self.one_hot_encode is True:
            self.convert_one_hot_encoding(self.test.fine_labels, data_type='test', class_type='fine')
            self.convert_one_hot_encoding(self.test.coarse_labels, data_type='test', class_type='coarse')
        if self.save_h5py != '':
            h5f = h5py.File(self.save_h5py, 'a')
            h5f.create_dataset('test_dataset', data=self.test.data, compression="gzip", compression_opts=9)
            print('Written CIFAR 100 test dataset to file: %s' % self.save_h5py)
            h5f.close()
        del test_fine_labels
        del test_coarse_labels
        del test_images
        del preprocessed_images
        if self.make_image is True:
            del images
        print()
        return True

    def load_data(self, train=True, test=True, data_directory='/tmp/cifar100/'):
        """
        
        :param train: 
        :param test: 
        :param data_directory: 
        :return: 
        """
        print('Loading CIFAR 100 Dataset')
        start = time.time()
        self.download_and_extract_data(data_directory)
        if train is True:
            print('Loading %d train images' %self.num_train_images)
            if self.train_validate_split is not None:
                print('Loading %d validate images' % self.num_validate_images)
            self.load_train_data(data_directory=data_directory)
        if test is True:
            print('Loading %d test images' % self.num_test_images)
            self.load_test_data(data_directory=data_directory)
        end = time.time()
        print('Loaded CIFAR 100 Dataset in %.4f seconds' %(end-start))
        return True
