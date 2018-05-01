import numpy as np
import math as m
from skimage import filters, feature

class Image():

    def __init__(self, image):
        self.image = image

    def print_data(self):
        print(self.image)

    def get_image(self):
        return self.image

    def get_filtered_image(self):
        return filters.gaussian(self.image, 1)

    def get_outlined_image(self):
        return feature.canny(self.get_filtered_image())
