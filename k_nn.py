from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=10)

    def train(self, images, targets):
        self.classifier.fit(images, targets)

    def predict(self, images):
        predictions = []
        for prediction in self.classifier.predict_proba(images):
            argmax = np.argmax(prediction)
            predictions.append([argmax, prediction[argmax]])
        return predictions

    def score(self, images, targets):
        return self.classifier.score(images, targets)