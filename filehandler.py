import matplotlib.image as img
from image import Image as dimg
import numpy as np

letter_path = "./data/chars74k-lite/"


def read_letter_images(letter):
    letter_images = []
    try:
        i = 0
        while True:
            image = dimg(img.imread(letter_path + letter + "/" + letter + "_" + i.__str__() + ".jpg"))
            letter_images.append(image)
            i += 1
    except FileNotFoundError:
        pass
    return np.array(letter_images)