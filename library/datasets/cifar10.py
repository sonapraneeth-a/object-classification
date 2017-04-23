from library.utils import file_utils
import os, shutil, time, matplotlib
import numpy as np
from scipy.misc import toimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from library.datasets.dataset import Dataset
from library.preprocessing.data_transform import transform


class CIFAR10:

    def __init__(self, num_images=1.0,
                 one_hot_encode=False,
                 train_validate_split=None,
                 preprocess='',
                 augment=False,
                 num_test_images=1.0,
                 endian='big',
                 make_image=True,
                 image_mode='rgb',
                 verbose=False):
        self.verbose = verbose
        self.img_height = 32
        self.img_width = 32
        self.num_channels = 3
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = 10
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
        self.file_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.file_md5 = 'c58f30108f718f92721af3b95e74349a'
        self.train = Dataset()
        self.validate = Dataset()
        self.test = Dataset()
        self.make_image = make_image
        self.image_mode = image_mode
        self.preprocess = preprocess
        self.augment = augment

    def get_image_width(self):
        return self.img_width

    def get_image_height(self):
        return self.img_height

    def get_num_image_channels(self):
        return self.img_channels

    def download_and_extract_data(self, data_directory):
        print('Downloading and extracting CIFAR 10 file')
        ## Step 1: Make the directories './datasets/cifar10/' if they do not exist
        if not os.path.exists(data_directory):
            if self.verbose is True:
                print('Creating the directory \'%s\'' % data_directory)
            file_utils.mkdir_p(data_directory)
        else:
            if self.verbose is True:
                print('Directory \'%s\' already exists' % data_directory)
        ## Step 2: Check if './datasets/cifar10/cifar-10-python.tar.gz' exists
        tar_file = data_directory + 'cifar-10.tar.gz'
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
        ## Step 3: Download CIFAR 10 dataset from 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        if make_tar is True:
            result = file_utils.download(self.file_url, tar_file, verbose=self.verbose)
            if result is False:
                if self.verbose is True:
                    raise FileNotFoundError('Download of CIFAR 10 dataset failed')
                return False
            result = file_utils.verify_md5(tar_file, self.file_md5, verbose=self.verbose)
            if result is False:
                if self.verbose is True:
                    raise FileNotFoundError('Downloaded CIFAR 10 dataset failed md5sum check')
                return False
        ## Step 4: Extract the datas
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
                    print('Extraction of CIFAR 10 dataset failed')
                return False
            else:
                if self.verbose is True:
                    print('Extraction of CIFAR 10 dataset success')
        return True

    def dict_read(self, dict_file):
        labels = dict_file[b'labels']
        data = dict_file[b'data']
        batch_label = dict_file[b'batch_label']
        filenames = dict_file[b'filenames']
        if self.verbose is True:
            print(dict_file.keys())
            print(labels[:5])
            print(batch_label[:5])
            print(filenames[:5])
            print(data[0].shape)
        return data, labels, batch_label, filenames

    def convert_images(self, raw, type='rgb'):
        raw_float = np.array(raw, dtype=float)
        if type == 'rgb_float':
            # Convert the raw images from the data-files to floating-points.
            raw_float = np.array(raw, dtype=float) / 255.0
        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, 3, 32, 32])
        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])
        images_n = images[0, :]
        if type == 'grey':
            images_n = ((images_n[:, :, 0] * 0.299) +
                        (images_n[:, :, 1] * 0.587) +
                        (images_n[:, :, 2] * 0.114))
        elif type == 'grey_float':
            images_n = ((images_n[:, :, 0] * 0.299) +
                        (images_n[:, :, 1] * 0.587) +
                        (images_n[:, :, 2] * 0.114))
            images_n = images_n / 255.0
        return images_n

    def convert_one_hot_encoding(self, classes, data_type='train'):
        num_classes = np.max(classes) + 1
        if data_type == 'train':
            self.train.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'validate':
            self.validate.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'test':
            self.test.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        for i in range(classes.shape[0]):
            if self.endian == 'big':
                if data_type == 'train':
                    self.train.one_hot_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_labels[i, num_classes-1-classes[i]] = 1
            if self.endian == 'little':
                if data_type == 'train':
                    self.train.one_hot_labels[i, classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_labels[i, classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_labels[i, classes[i]] = 1

    def load_train_data(self, split=False, data_directory='/tmp/cifar10/'):
        print('Loading CIFAR 10 Training Dataset')
        basic_dir_path = data_directory + 'cifar-10-batches/'
        data_batch_path = 'data_batch_'
        data_files = []
        data_dict = []
        for i in range(1, 6):
            data_files.append(str(basic_dir_path + data_batch_path + str(i)))
        for file in data_files:
            # print('Unpickling data file: %s' % file)
            data_dict.append(file_utils.unpickle(file))
        data_labels = []
        data_images = []
        for i in range(len(data_dict)):
            print('Reading unpicked data file: %s' %data_files[i])
            data, labels, _, _ = self.dict_read(data_dict[i])
            data_labels.extend(labels)
            data_images.extend(data)
        data_images = np.array(data_images)
        data_labels = np.array(data_labels)
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
            self.train.class_labels = np.array(data_labels[:self.num_images])
            self.train.class_names = np.array(list(map(lambda x: self.classes[x], self.train.class_labels)))
        else:
            print('Requested to use only %d images' %self.num_images)
            self.train.data = np.array(preprocessed_images[:self.num_train_images, :])
            if self.make_image is True:
                self.train.images = np.array(images[:self.num_train_images, :])
            self.train.class_labels = np.array(data_labels[:self.num_train_images])
            self.train.class_names = np.array(list(map(lambda x: self.classes[x], self.train.class_labels)))
            self.validate.data = \
                np.array(preprocessed_images[self.num_train_images:self.num_train_images+self.num_validate_images, :])
            if self.make_image is True:
                self.validate.images = np.array(images[self.num_train_images:self.num_train_images+self.num_validate_images, :])
            self.validate.class_labels = \
                np.array(data_labels[self.num_train_images:self.num_train_images+self.num_validate_images])
            self.validate.class_names = np.array(list(map(lambda x: self.classes[x], self.validate.class_labels)))
        if self.one_hot_encode is True:
            self.convert_one_hot_encoding(self.train.class_labels, data_type='train')
            if self.train_validate_split is not None:
                self.convert_one_hot_encoding(self.validate.class_labels, data_type='validate')
        del data_labels
        del data_images
        del preprocessed_images
        if self.make_image is True:
            del images
        return True

    def load_test_data(self, data_directory='/tmp/cifar10/'):
        print('Loading CIFAR 10 Test Dataset')
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
        preprocessed_images = transform(test_images, transform_method=self.preprocess)
        if self.make_image is True:
            images = []
            for fig_num in range(preprocessed_images.shape[0]):
                fig = preprocessed_images[fig_num, :]
                img = self.convert_images(fig, type=self.image_mode)
                images.append(img)
            images = np.array(images)
        test_labels = np.array(test_labels)
        self.test.data = np.array(preprocessed_images[:self.num_test_images])
        if self.make_image is True:
            self.test.images = np.array(images[:self.num_test_images, :])
        self.test.class_labels = np.array(test_labels[:self.num_test_images])
        self.test.class_names = np.array(list(map(lambda x: self.classes[x], self.test.class_labels)))
        if self.one_hot_encode is True:
            self.convert_one_hot_encoding(self.test.class_labels, data_type='test')
        del test_labels
        del test_images
        del preprocessed_images
        if self.make_image is True:
            del images
        return True

    def load_data(self, train=True, test=True, data_directory='/tmp/cifar10/'):
        print('Loading CIFAR 10 Dataset')
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
        print('Loaded CIFAR 10 Dataset in %.4f seconds' %(end-start))
        return True

    def train_images(self):
        return self.train.data

    def train_labels(self):
        return self.train.one_hot_labels

    def train_classes(self):
        return self.train.class_labels

    def validate_images(self):
        return self.validate.data

    def validate_labels(self):
        return self.validate.one_hot_labels

    def validate_classes(self):
        return self.validate.class_labels

    def test_images(self):
        return self.test.data

    def test_labels(self):
        return self.test.one_hot_labels

    def test_classes(self):
        return self.test.class_labels

    def plot(self, grid, matrix, fontsize=10):
        k = 0
        class_type = 0
        for ax in grid:
            ax.imshow(toimage(matrix[k, :]))
            ax.title.set_visible(False)
            # ax.axis('tight')
            # ax.axis('off')
            ax.set_frame_on(False)
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlim([0, 32])
            ax.set_ylim([32, 0])
            if k % 10 == 0:
                ax.set_ylabel(self.classes[class_type], rotation=0, ha='right',
                              weight='bold', size=fontsize)
                class_type += 1
            k += 1
        plt.tight_layout()
        plt.show()

    def plot_sample(self, plot_data=True, plot_test=False, verbose=False, fig_size=(7, 7), fontsize=12,
                    images_per_class=10):
        num_images_per_class = images_per_class
        if self.train.data is None and plot_test is False:
            self.load_data(train=plot_data, test=plot_test)
        elif plot_data is False and self.test.data is None:
            self.load_data(train=plot_data, test=plot_test)
        elif self.train.data is None and self.test.data is None:
            self.load_data(train=plot_data, test=plot_test)
        data_image_nos = []
        test_image_nos = []
        if plot_data is True:
            for class_type in range(self.num_classes):
                data_class_labels = np.where(self.train.class_labels == class_type)
                data_class_labels = data_class_labels[0][:num_images_per_class]
                data_class_labels = data_class_labels.tolist()
                data_image_nos.extend(data_class_labels)
            example_data_images = self.train.data[data_image_nos, :]
            example_data_image_matrix = []
            k = 0
            for i in range(len(data_image_nos)):
                example_data_image_matrix.append(self.convert_images(example_data_images[k, :]))
                k += 1
            example_data_image_matrix = np.array(example_data_image_matrix)
        if plot_test is True:
            for class_type in range(self.num_classes):
                test_class_labels = np.where(self.test.class_labels == class_type)
                test_class_labels = test_class_labels[0][:num_images_per_class]
                test_class_labels = test_class_labels.tolist()
                test_image_nos.extend(test_class_labels)
            example_test_images = self.test.data[test_image_nos, :]
            example_test_image_matrix = []
            k = 0
            for i in range(len(test_image_nos)):
                example_test_image_matrix.append(self.convert_images(example_test_images[k, :]))
                k += 1
            example_test_image_matrix = np.array(example_test_image_matrix)
        num_rows = 10
        num_cols = num_images_per_class
        if verbose is True:
            print('Plot image matrix shape: ' + str(example_data_image_matrix.shape))
            print('Number of rows: %d' % num_rows)
            print('Number of cols: %d' % num_cols)
        if plot_data is True:
            print('Plotting CIFAR 10 Train Dataset')
            data_fig = plt.figure()
            data_fig.set_figheight(fig_size[0])
            data_fig.set_figwidth(fig_size[1])
            data_grid = Grid(data_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(data_grid, example_data_image_matrix, fontsize=fontsize)
        if plot_test is True:
            print('Plotting CIFAR 10 Test Dataset')
            test_fig = plt.figure()
            test_fig.set_figheight(fig_size[0])
            test_fig.set_figwidth(fig_size[1])
            test_grid = Grid(test_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(test_grid, example_test_image_matrix, fontsize=fontsize)

    def plot_images(self, images, cls_true, cls_pred=None, nrows=3, ncols=3, fig_size=(7,7),
                    fontsize=15, convert=False, type='rgb'):
        assert images.shape[0] == cls_true.shape[0]
        fig, axes = plt.subplots(nrows, ncols)
        if fig_size is not None:
            fig.set_figheight(fig_size[0])
            fig.set_figwidth(fig_size[1])
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for image_no, ax in enumerate(axes.flat):
            # Plot image.
            if convert is True:
                image = self.convert_images(images[image_no, :])
            else:
                image = images[image_no, :]
            if type == 'rgb':
                ax.imshow(toimage(image), cmap='binary')
            if type == 'grey':
                ax.imshow(toimage(image), cmap=matplotlib.cm.Greys_r)
            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[image_no])
            else:
                xlabel = "True: {0}\nPred: {1}".format(cls_true[image_no], cls_pred[image_no])
            ax.set_xlabel(xlabel, weight='bold', size=fontsize)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return True

