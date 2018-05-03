from imagehandler import ImageHandler
import extractor as e
from image import Image as dimg
import matplotlib.image as img
from k_nn import KNN
from svm import SVM
from copy import deepcopy
from matplotlib import pyplot as plt


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[0] - window_size[0] + 1, step_size):
            yield(x, y, image[y:y + window_size[1], x:x + window_size[0]])


image_handler = ImageHandler()
train_images, train_targets = image_handler.get_all_train_data()
test_images, test_targets = image_handler.get_all_test_data()
method = [e.apply_gaussian_filter, e.calculate_hog]
train_images = e.extract(method, train_images, True, True, 30)
test_images = e.extract(method, test_images, True)
k_nn = KNN()
svm = KNN()
k_nn.train(train_images, train_targets)
svm.train(train_images, train_targets)
image = dimg(img.imread("data/detection-images/detection-2.jpg"), None).get_image()
frames = []
for frame in sliding_window(image, 1, [20, 20]):
    add = True
    y_white = 0
    x_white = 0
    for index, i in enumerate(frame[2]):
        for index2, j in enumerate(i):
            if frame[2][index][index2] == 255:
                x_white += 1
            else:
                x_white = 0
            if frame[2][index2][index] == 255:
                y_white += 1
            else:
                y_white = 0
            if y_white >= 400/40 or x_white >= 400/40:
                add = False
                break
        if not add:
            break
    if add:
        frames.append(frame[2])
extracted_frames = e.extract(method, deepcopy(frames), True)
predictions = svm.predict(extracted_frames)
for i, prediction in enumerate(predictions):
    if prediction[1] > 0.5:
        print(ImageHandler.alphabet[prediction[0]])
        #plt.imshow(frames[i], cmap='gray')
        #plt.show()