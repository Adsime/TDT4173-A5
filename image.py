import numpy as np
import math as m


class Image:
    def __init__(self, image, target):
        self.image = image
        self.target = target

    def get_image(self):
        return self.image

    def get_target(self):
        return self.target
