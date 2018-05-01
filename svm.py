from sklearn.svm import SVC


class SVM:
    def __init__(self):
        self.classifier = SVC()

    def train(self, training_data, targets):
        self.classifier.fit(training_data, targets)

    def predict(self, prediction_data):
        return self.classifier.predict(prediction_data)
