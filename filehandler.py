import matplotlib.image as img
from image import Image as dimg
import numpy as np

letter_path = "./data/chars74k-lite/"
detection_path = "./data/detection-images/detection-"
extension = ".jpg"


def read_letter_images(letter):
    letter_images = []
    try:
        i = 0
        while True:
            image = dimg(img.imread(letter_path + letter + "/" + letter + "_" + i.__str__() + extension), letter)
            letter_images.append(image)
            i += 1
    except FileNotFoundError:
        pass
    print("Images for " + letter + " loaded successfully")
    return np.array(letter_images)


def read_detection_image(identifier):
    return img.imread(detection_path + identifier.__str__() + extension)