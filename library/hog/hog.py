import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
import cv2


class HOG:

    def __init__(self, block_size=(8,8), cell_size=(2,2), nbins=9):
        self.block_size = block_size
        self.cell_size = cell_size
        self.nbins = nbins

    def make_hog_gradients(self, img):
        # winSize is the size of the image cropped to an multiple of the cell size
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // self.cell_size[1] * self.cell_size[1],
                                          img.shape[0] // self.cell_size[0] * self.cell_size[0]),
                                _blockSize=(self.block_size[1] * self.cell_size[1],
                                            self.block_size[0] * self.cell_size[0]),
                                _blockStride=(self.cell_size[1], self.cell_size[0]),
                                _cellSize=(self.cell_size[1], self.cell_size[0]),
                                _nbins=self.nbins)
        n_cells = (img.shape[0] // self.cell_size[0], img.shape[1] // self.cell_size[1])
        hog_feats = hog.compute(img) \
            .reshape(n_cells[1] - self.block_size[1] + 1,
                     n_cells[0] - self.block_size[0] + 1,
                     self.block_size[0], self.block_size[1], self.nbins) \
            .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
        # hog_feats now contains the gradient amplitudes for each direction,
        # for each cell of its group for each group. Indexing is by rows then columns.
        gradients = np.zeros((n_cells[0], n_cells[1], self.nbins))
        # count cells (border cells appear less often across overlapping groups)
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
        for off_y in range(self.block_size[0]):
            for off_x in range(self.block_size[1]):
                gradients[off_y:n_cells[0] - self.block_size[0] + off_y + 1,
                off_x:n_cells[1] - self.block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - self.block_size[0] + off_y + 1,
                off_x:n_cells[1] - self.block_size[1] + off_x + 1] += 1
        # Average gradients
        gradients /= cell_count
        return gradients
