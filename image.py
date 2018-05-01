import numpy as np
import math as m

class Image():

    def __init__(self, image):
        self.image = image

    def print_data(self):
        print(self.image)

    def get_image_map(self):
        return self.image

    def smooth(self):
        new_img = np.zeros((20, 20))
        for x, row in enumerate(self.image):
            for y, elem in enumerate(row):
                new_img[x][y] = 0 if elem < 75 else 255 if elem > 170 else 125
        return new_img
