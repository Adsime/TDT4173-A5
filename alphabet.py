import numpy as np
import matplotlib.image as img
from image import Image as dimg

class Alphabet:
    letter_path = "./data/chars74k-lite/"
    alphabet = np.array(['a', 'b', 'c', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    data = None

    def __init__(self):
        self.create_map()

    def read_data(self, letter):
        letter_images = []
        try:
            i = 0
            while True:
                image = dimg(img.imread(self.letter_path + letter + "/" + letter + "_" + i.__str__() + ".jpg"))
                letter_images.append(image)
                i += 1
        except FileNotFoundError:
            pass
        return np.array(letter_images)

    def create_map(self):
        data = []
        for letter in self.alphabet:
            data.append(self.read_data(letter))
        self.data = np.array(data)

    def get_images(self, letter):
        return self.data[np.where(self.alphabet == letter)[0][0]]
