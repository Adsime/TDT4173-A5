from sklearn.svm import SVC
import numpy as np

class SVM:
    def __init__(self):
        self.classifier = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
                              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                              max_iter=-1, probability=True, random_state=None, shrinking=True,
                              tol=0.001, verbose=False)

    def train(self, images, targets):
        self.classifier.fit(images, targets)

    def predict(self, images):
        predictions = []
        for prediction in self.classifier.predict(images):
            argmax = np.argmax(prediction)
            predictions.append([argmax, prediction[argmax]])
        return predictions

    def score(self, images, targets):
        return self.classifier.score(images, targets)
