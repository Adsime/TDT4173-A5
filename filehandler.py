import matplotlib.image as img
from image import Image as dimg


def read_letter_images(letter):
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