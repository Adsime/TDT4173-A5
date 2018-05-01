import numpy as np
import sklearn.model_selection as tt
import filehandler as fh

class Alphabet:

    # Data fields
    alphabet = np.array(['a', 'b', 'c', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    data = None
    train_data = None
    test_data = None

    # Class methods for instantiation

    def __init__(self, test_sample_size=0.25):
        self.create_map()
        self.create_test_train(test_sample_size)

    def create_map(self):
        data = []
        for letter in self.alphabet:
            data.append(fh.read_letter_images(letter))
        self.data = np.array(data)

    def create_test_train(self, test_sample_size):
        train_data = []
        test_data = []
        for letter in self.data:
            x, y = tt.train_test_split(letter, test_size=test_sample_size)
            train_data.append(x)
            test_data.append(y)
        self.train_data = train_data
        self.test_data = test_data

    # Callable methods

    def get_images(self, letter):
        return self.data[np.where(self.alphabet == letter)[0][0]]

    def get_test_set(self, letter):
        return self.test_data[np.where(self.alphabet == letter)[0][0]]

    def get_train_set(self, letter):
        return self.train_data[np.where(self.alphabet == letter)[0][0]]

    def get_all_train_data(self):
        return self.train_data

    def get_all_test_data(self):
        return self.test_data