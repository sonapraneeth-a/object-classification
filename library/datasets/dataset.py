class Dataset:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.images = None
        self.orig_data = None
        self.preprocess_data = None
        self.augment_data = None
        self.one_hot_labels = None
        self.coarse_labels = None
        self.fine_labels = None
        self.filenames = None
        self.info = None
        self.class_names = None

    def get_next_batch(self, batch_size=100):
        return True