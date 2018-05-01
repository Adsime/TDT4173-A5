from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self):
        self.classifier = NearestNeighbors()

    def train(self, training_data, targets):
        self.classifier.fit(training_data, targets)

    def predict(self, prediction_data):
        return self.classifier.predict(prediction_data)
