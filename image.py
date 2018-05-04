import numpy as np
import math as m
from copy import deepcopy


class Image:
    def __init__(self, image, target):
        self.image = image
        self.target = target

    def get_image(self):
        return self.image

    def get_target(self):
        return self.target

    def rot90(self):
        return Image(np.rot90(self.image), self.target)

    def invert(self):
        return Image(255 - self.image, self.rot90())

    def mirror(self):
        return Image(np.flip(self.image, 1), self.target)
