from image_handler import Alphabet
import extractor as e
from k_nn import KNN
from svm import SVM

alphabet = Alphabet()

train_images, train_targets = alphabet.get_all_train_data()
test_images, test_targets = alphabet.get_all_test_data()

methods = [e.apply_gaussian_filter]

train_images = e.extract(methods, train_images)
test_images = e.extract(methods, test_images)

knn = KNN()
svm = SVM()

knn.train(train_images, train_targets)
svm.train(train_images, train_targets)

print(knn.score(test_images, test_targets))
print(svm.score(test_images, test_targets))
