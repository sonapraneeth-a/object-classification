import matplotlib
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from library.datasets.dataset import Dataset


class CIFARBase:

    def __init__(self,
                 num_images=1.0,
                 one_hot_encode=False,
                 train_validate_split=None,
                 preprocess='',
                 augment=False,
                 num_test_images=1.0,
                 endian='little',
                 make_image=True,
                 image_mode='rgb',
                 save_h5py=True,
                 verbose=False):
        """
        
        :param num_images: 
        :param one_hot_encode: 
        :param train_validate_split: 
        :param preprocess: 
        :param augment: 
        :param num_test_images: 
        :param endian: 
        :param make_image: 
        :param image_mode: 
        :param save_h5py: 
        :param verbose: 
        """
        self.verbose = verbose
        self.img_height = 32
        self.img_width = 32
        self.num_channels = 3
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
        self.train = Dataset()
        self.validate = Dataset()
        self.test = Dataset()
        self.make_image = make_image
        self.image_mode = image_mode
        self.preprocess = preprocess
        self.augment = augment
        self.save_h5py = save_h5py
        self.fine_classes = None
        self.coarse_classes = None
        self.num_fine_classes = None
        self.num_coarse_classes = None
        self.file_url = None
        self.file_md5 = None

    def get_image_width(self):
        """
        
        :return: 
        """
        return self.img_width

    def get_image_height(self):
        """
        
        :return: 
        """
        return self.img_height

    def get_num_image_channels(self):
        """
        
        :return: 
        """
        return self.img_channels

    def convert_images(self, raw, type='rgb'):
        """
        
        :param raw: 
        :param type: 
        :return: 
        """
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

    def convert_one_hot_encoding(self, classes, data_type='train', class_type='fine'):
        """
        
        :param classes: 
        :param data_type: 
        :param class_type: 
        :return: 
        """
        num_classes = np.max(classes) + 1
        if data_type == 'train' and class_type == 'fine':
            self.train.one_hot_fine_labels = np.zeros((classes.shape[0], num_classes))
        elif data_type == 'train' and class_type == 'coarse':
            self.train.one_hot_coarse_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'validate' and class_type == 'fine':
            self.validate.one_hot_fine_labels = np.zeros((classes.shape[0], num_classes))
        elif data_type == 'validate' and class_type == 'coarse':
            self.validate.one_hot_coarse_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'test' and class_type == 'fine':
            self.test.one_hot_fine_labels = np.zeros((classes.shape[0], num_classes))
        elif data_type == 'test' and class_type == 'coarse':
            self.test.one_hot_coarse_labels = np.zeros((classes.shape[0], num_classes))

        for i in range(classes.shape[0]):
            if self.endian == 'big' and class_type == 'fine':
                if data_type == 'train':
                    self.train.one_hot_fine_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_fine_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_fine_labels[i, num_classes-1-classes[i]] = 1
            elif self.endian == 'big' and class_type == 'coarse':
                if data_type == 'train':
                    self.train.one_hot_coarse_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_coarse_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_coarse_labels[i, num_classes-1-classes[i]] = 1
            if self.endian == 'little' and class_type == 'fine':
                if data_type == 'train':
                    self.train.one_hot_fine_labels[i, classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_fine_labels[i, classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_fine_labels[i, classes[i]] = 1
            elif self.endian == 'little' and class_type == 'coarse':
                if data_type == 'train':
                    self.train.one_hot_coarse_labels[i, classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_coarse_labels[i, classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_coarse_labels[i, classes[i]] = 1

    def train_images(self):
        """
        
        :return: 
        """
        return self.train.data

    def train_labels(self):
        """
        
        :return: 
        """
        return self.train.one_hot_labels

    def train_classes(self):
        """
        
        :return: 
        """
        return self.train.fine_labels

    def validate_images(self):
        """
        
        :return: 
        """
        return self.validate.data

    def validate_labels(self):
        """
        
        :return: 
        """
        return self.validate.one_hot_labels

    def validate_classes(self):
        """
        
        :return: 
        """
        return self.validate.fine_labels

    def test_images(self):
        """
        
        :return: 
        """
        return self.test.data

    def test_labels(self):
        """
        
        :return: 
        """
        return self.test.one_hot_labels

    def test_classes(self):
        """
        
        :return: 
        """
        return self.test.fine_labels

    def plot(self, grid, matrix, fontsize=10):
        """
        
        :param grid: 
        :param matrix: 
        :param fontsize: 
        :return: 
        """
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
                ax.set_ylabel(self.fine_classes[class_type], rotation=0, ha='right',
                              weight='bold', size=fontsize)
                class_type += 1
            k += 1
        plt.tight_layout()
        plt.show()

    def plot_sample(self, plot_train=True, plot_test=False, verbose=False, fig_size=(7, 7), fontsize=12,
                    images_per_class=10):
        """
        
        :param plot_train: 
        :param plot_test: 
        :param verbose: 
        :param fig_size: 
        :param fontsize: 
        :param images_per_class: 
        :return: 
        """
        num_images_per_class = images_per_class
        if self.train.data is None and plot_test is False:
            self.load_data(train=plot_train, test=plot_test)
        elif plot_train is False and self.test.data is None:
            self.load_data(train=plot_train, test=plot_test)
        elif self.train.data is None and self.test.data is None:
            self.load_data(train=plot_train, test=plot_test)
        data_image_nos = []
        test_image_nos = []
        if plot_train is True:
            for class_type in range(self.num_fine_classes):
                data_fine_labels = np.where(self.train.fine_labels == class_type)
                data_fine_labels = data_fine_labels[0][:num_images_per_class].tolist()
                data_image_nos.extend(data_fine_labels)
            example_data_images = self.train.data[data_image_nos, :]
            example_data_image_matrix = []
            k = 0
            for i in range(len(data_image_nos)):
                example_data_image_matrix.append(self.convert_images(example_data_images[k, :]))
                k += 1
            example_data_image_matrix = np.array(example_data_image_matrix)
        if plot_test is True:
            for class_type in range(self.num_fine_classes):
                test_fine_labels = np.where(self.test.fine_labels == class_type)
                test_fine_labels = test_fine_labels[0][:num_images_per_class]
                test_fine_labels = test_fine_labels.tolist()
                test_image_nos.extend(test_fine_labels)
            example_test_images = self.test.data[test_image_nos, :]
            example_test_image_matrix = []
            k = 0
            for i in range(len(test_image_nos)):
                example_test_image_matrix.append(self.convert_images(example_test_images[k, :]))
                k += 1
            example_test_image_matrix = np.array(example_test_image_matrix)
        num_rows = min(self.num_fine_classes, 20)
        num_cols = num_images_per_class
        if verbose is True:
            print('Plot image matrix shape: ' + str(example_data_image_matrix.shape))
            print('Number of rows: %d' % num_rows)
            print('Number of cols: %d' % num_cols)
        if plot_train is True:
            data_fig = plt.figure()
            data_fig.set_figheight(fig_size[0])
            data_fig.set_figwidth(fig_size[1])
            data_grid = Grid(data_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(data_grid, example_data_image_matrix, fontsize=fontsize)
        if plot_test is True:
            test_fig = plt.figure()
            test_fig.set_figheight(fig_size[0])
            test_fig.set_figwidth(fig_size[1])
            test_grid = Grid(test_fig, rect=111, nrows_ncols=(num_rows, num_cols),
                             axes_pad=0.0, label_mode='R',
                             )
            self.plot(test_grid, example_test_image_matrix, fontsize=fontsize)

    def plot_images(self, images, cls_true_fine, cls_true_coarse=None, cls_pred_fine=None, cls_pred_coarse=None,
                    nrows=3, ncols=3, fig_size=(7, 7), fontsize=15, convert=False, type='rgb'):
        """
        
        :param images: 
        :param cls_true_fine: 
        :param cls_true_coarse: 
        :param cls_pred_fine: 
        :param cls_pred_coarse: 
        :param nrows: 
        :param ncols: 
        :param fig_size: 
        :param fontsize: 
        :param convert: 
        :param type: 
        :return: 
        """
        assert images.shape[0] == cls_true_fine.shape[0]
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
            if cls_pred_fine is None and cls_pred_coarse is None:
                if cls_true_coarse is None:
                    xlabel = "Fine: {0}".format(cls_true_fine[image_no])
                else:
                    xlabel = "Fine: {0}\nCoarse: {1}".format(cls_true_fine[image_no], cls_true_coarse[image_no])
            else:
                if cls_true_coarse is None:
                    if cls_pred_fine is None:
                        xlabel = "Fine: {0}".format(cls_true_fine[image_no])
                    else:
                        xlabel = "Fine: {0}\nPred. Fine: {1}".format(cls_true_fine[image_no], cls_pred_fine[image_no])
                else:
                    if cls_pred_coarse is None:
                        xlabel = "Coarse: {0}".format(cls_true_coarse[image_no])
                    else:
                        xlabel = "Coarse: {0}\nPred. Coarse: {1}".format(cls_true_coarse[image_no],
                                                                         cls_pred_coarse[image_no])
            ax.set_xlabel(xlabel, weight='bold', size=fontsize)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return True

