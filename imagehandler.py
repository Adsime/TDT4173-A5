import numpy as np
import sklearn.model_selection as tt
import filehandler as fh


class ImageHandler:

    # Data fields
    alphabet = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    data = None
    train_images = None
    test_images = None

    # Class methods for instantiation

    def __init__(self, test_sample_size=0.20):
        self.create_map()
        self.create_test_train(test_sample_size)

    def create_map(self):
        data = []
        for letter in self.alphabet:
            data.append(fh.read_letter_images(letter))
        self.data = np.array(data)

    def create_test_train(self, test_sample_size):
        train_images = []
        test_images = []
        for letter in self.data:
            x, y = tt.train_test_split(letter, test_size=test_sample_size)
            train_images.append(x)
            test_images.append(y)
        self.train_images = train_images
        self.test_images = test_images
        print("Training set loaded successfully, " + (len(self.train_images)).__str__() + " samples")
        print("Test set loaded successfully, " + (len(self.test_images)).__str__() + " samples")
    # Callable methods

    def get_images(self, letter):
        return self.data[np.where(self.alphabet == letter)[0][0]]

    def get_test_set(self, letter):
        return self.test_images[np.where(self.alphabet == letter)[0][0]]

    def get_train_set(self, letter):
        return self.train_images[np.where(self.alphabet == letter)[0][0]]

    def get_all_train_data(self):
        return prepare_data(self.train_images)

    def get_all_test_data(self):
        return prepare_data(self.test_images)


def prepare_data(images):
    data = []
    targets = []
    for letter in images:
        for img in letter:
            targets.append(img.get_target())
            data.append(img)
    return [data, targets]

