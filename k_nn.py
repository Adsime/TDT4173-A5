from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5)

    def train(self, images, targets):
        print("Training KNN")
        self.classifier.fit(images, targets)
        print("KNN trained successfully")

    def predict(self, images):
        predictions = []
        for prediction in self.classifier.predict_proba(images):
            argmax = np.argmax(prediction)
            predictions.append([argmax, prediction[argmax]])
        print("KNN finished predicting " + (len(images)).__str__() + " samples")
        return predictions

    def score(self, images, targets):
        return self.classifier.score(images, targets)