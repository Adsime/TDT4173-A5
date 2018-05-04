from matplotlib import pyplot as plt
from imagehandler import ImageHandler
import matplotlib.patches as patches
import numpy as np


class Detector:

    def __init__(self, image, step_size, window_size):
        self.image = image
        self.step_size = step_size
        self.window_size = window_size

    def sliding_window(self):
        print("Iterating image:")
        x_range = self.image.shape[1]
        y_range = self.image.shape[0]
        window_coordinates = []
        windows = []
        for y in range(0, y_range - self.window_size + 1, self.step_size):
            print("{0:.2f}% done".format(y*100/(y_range - self.window_size)))
            for x in range(0, x_range - self.window_size, self.step_size):
                window = self.image[y: y + self.window_size, x: x + self.window_size]
                white_space = 0
                for i in range(self.window_size):
                    white_row = True
                    white_column = True
                    for j in range(self.window_size):
                        if window[j][i] <= 250:
                            white_row = False
                        if window[i][j] <= 250:
                            white_column = False
                    if white_row:
                        white_space += 1
                    if white_column:
                        white_space += 1
                if white_space < 3:
                    added = False
                    for i, window_x_y in enumerate(window_coordinates):
                        if abs(window_x_y[0] - x) < 6 and abs(window_x_y[1] - y) < 6:
                            window_coordinates[i] = [x, y]
                            windows[i].append(window)
                            added = True
                            break
                    if not added:
                        window_coordinates.append([x, y])
                        windows.append([window])
        print("Images with much whitespace removed")
        print("Similar images grouped")
        self.window_coordinates = window_coordinates
        res = []
        n_equal_elements = []
        for equal_windows in windows:
            n = 0
            for equal_window in equal_windows:
                n += 1
                res.append(np.array(equal_window))
            n_equal_elements.append(n)
        self.n_equal_elements = n_equal_elements
        return res

    def show_results(self, predictions, threshold):
        fig1 = plt.figure()
        i = 0
        for iter, letters in enumerate(self.n_equal_elements):
            highest_prob = -1
            index = -1
            for k, prediction in enumerate(predictions[i:i + letters]):
                if highest_prob < prediction[1]:
                    highest_prob = prediction[1]
                    index = k
            if highest_prob > threshold:
                ax = fig1.add_subplot(111, aspect='equal')
                ax.add_patch(patches.Rectangle((self.window_coordinates[iter][0], self.window_coordinates[iter][1]), 20,
                                               20, fill=False, edgecolor='red'))
                ax.text(self.window_coordinates[iter][0], self.window_coordinates[iter][1] - 5,
                        ImageHandler.alphabet[predictions[i + index][0]], color='red', fontsize=20)
            i += letters
        plt.imshow(self.image, cmap='gray')
        plt.show()



