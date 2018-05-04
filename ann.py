from sklearn.neural_network import MLPClassifier as ann
import numpy as np


class ANN:
    def __init__(self):
        self.classifier = ann()

    def train(self, images, targets):
        self.classifier.fit(images, targets)
        print("SVM trained successfully")

    def predict(self, images):
        predictions = []
        for prediction in self.classifier.predict(images):
            argmax = np.argmax(prediction)
            predictions.append([argmax, prediction[argmax]])
        print("SVM finished predicting " + (len(images)).__str__() + " samples")
        return predictions

    def score(self, images, targets):
        return self.classifier.score(images, targets)