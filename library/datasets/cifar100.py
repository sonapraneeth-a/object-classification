from library.utils import file_utils
import os, shutil, time, matplotlib
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from library.datasets.dataset import Dataset
from library.preprocessing.data_transform import transform


class CIFAR100:

    def __init__(self, num_images=1.0,
                 one_hot_encode=False,
                 train_validate_split=None,
                 preprocess='',
                 augment=False,
                 num_test_images=1.0,
                 endian='little',
                 make_image=True,
                 image_mode='rgb',
                 verbose=False):
        self.verbose = verbose
        self.img_height = 32
        self.img_width = 32
        self.num_channels = 3
        self.num_fine_classes = 20
        self.num_coarse_classes = 100
        self.one_hot_encode = one_hot_encode
        self.endian = endian
        self.train_validate_split = train_validate_split
        if num_images > 1.0 or num_images < 0.0:
            self.num_images = int(50000)
        else:
            self.num_images = int(num_images*50000)
        if self.train_validate_split is not None:
            self.num_train_images = int(self.train_validate_split*self.num_images)
            self.num_validate_images = self.num_images - self.num_train_images
        else:
            self.num_train_images = int(self.num_images)
        if num_test_images > 1.0 or num_test_images < 0.0:
            self.num_test_images = int(10000)
        else:
            self.num_test_images = int(num_test_images*10000)
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
        self.train = Dataset()
        self.validate = Dataset()
        self.test = Dataset()
        self.make_image = make_image
        self.image_mode = image_mode
        self.preprocess = preprocess
        self.augment = augment

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

    def convert_images(self, raw, type='rgb'):
        if type == 'float':
            # Convert the raw images from the data-files to floating-points.
            raw_float = np.array(raw, dtype=float) / 255.0
        elif type == 'rgb':
            raw_float = np.array(raw, dtype=float)
        else:
            print('Unknown format')
        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, 3, 32, 32])
        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])
        images_n = images[0, :]
        return images_n

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
            data, labels, _, _ = self.dict_read(data_dict[i])
            data_labels.extend(labels)
            data_images.extend(data)
        self.data['train_images'] = np.array(data_images)
        self.data['train_labels'] = np.array(data_labels)
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

    def plot(self, grid, matrix, fontsize=10, fine=True, coarse=False):
        k = 0
        class_type = 0
        for ax in grid:
            ax.imshow(toimage(matrix[k, :]))
            ax.title.set_visible(False)
            ax.axis('tight')
            # ax.axis('off')
            ax.set_frame_on(False)
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlim([0, 32])
            ax.set_ylim([32, 0])
            if k % 10 == 0:
                ax.set_ylabel(self.fine_classes[class_type], rotation=0, ha='right',
                              weight='bold', size=fontsize)
                class_type += 1
            k += 1
        plt.tight_layout()
        plt.show()

    def plot_sample(self, plot_data=True, plot_test=False, verbose=False, fig_size=(10, 9.5), fontsize=10,
                    fine=True, coarse=False, images_per_class=10):
        num_images_per_class = images_per_class
        if self.data['train_images'] is None and plot_test is False:
            self.load_data(train=plot_data, test=plot_test)
        elif plot_data is False and self.data['test_images'] is None:
            self.load_data(train=plot_data, test=plot_test)
        elif self.data['train_images'] is None and self.data['test_images'] is None:
            self.load_data(train=plot_data, test=plot_test)
        data_image_nos = []
        test_image_nos = []
        if plot_data is True:
            for class_type in range(self.num_fine_classes):
                data_class_labels = np.where(self.data['train_labels'] == class_type)
                data_class_labels = data_class_labels[0][:num_images_per_class]
                data_class_labels = data_class_labels.tolist()
                data_image_nos.extend(data_class_labels)
            example_data_images = self.data['train_images'][data_image_nos, :]
            example_data_image_matrix = []
            k = 0
            for i in range(len(data_image_nos)):
                example_data_image_matrix.append(self.convert_images(example_data_images[k, :]))
                k += 1
            example_data_image_matrix = np.array(example_data_image_matrix)
        if plot_test is True:
            for class_type in range(self.num_fine_classes):
                test_class_labels = np.where(self.data['test_labels'] == class_type)
                test_class_labels = test_class_labels[0][:num_images_per_class]
                test_class_labels = test_class_labels.tolist()
                test_image_nos.extend(test_class_labels)
            example_test_images = self.data['test_images'][test_image_nos, :]
            example_test_image_matrix = []
            k = 0
            for i in range(len(test_image_nos)):
                example_test_image_matrix.append(self.convert_images(example_test_images[k, :]))
                k += 1
            example_test_image_matrix = np.array(example_test_image_matrix)
        num_rows = self.num_fine_classes
        num_cols = num_images_per_class
        if verbose is True:
            print('Plot image matrix shape: ' + str(example_data_image_matrix.shape))
            print('Number of rows: %d' % num_rows)
            print('Number of cols: %d' % num_cols)
        if plot_data is True:
            print('Plotting CIFAR 100 Train Dataset')
            data_fig = plt.figure()
            data_fig.set_figheight(fig_size[0])
            data_fig.set_figwidth(fig_size[1])
            data_grid = Grid(data_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(data_grid, example_data_image_matrix, fontsize=fontsize)
        if plot_test is True:
            print('Plotting CIFAR 100 Test Dataset')
            test_fig = plt.figure()
            test_fig.set_figheight(fig_size[0])
            test_fig.set_figwidth(fig_size[1])
            test_grid = Grid(test_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(test_grid, example_test_image_matrix, fontsize=fontsize)

