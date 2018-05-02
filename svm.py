from sklearn.svm import SVC


class SVM:
    def __init__(self):
        self.classifier = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
                              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                              max_iter=-1, probability=False, random_state=None, shrinking=True,
                              tol=0.001, verbose=False)

    def train(self, training_data, targets):
        self.classifier.fit(training_data, targets)

    def predict(self, prediction_data):
        return self.classifier.predict(prediction_data)
