from sklearn.svm import SVC
import numpy as np

class SVM:
    def __init__(self):
        self.classifier = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
                              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                              max_iter=-1, probability=True, random_state=None, shrinking=True,
                              tol=0.001, verbose=False)

        self.classifier = SVC(C=950.0, gamma=0.00095)

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
