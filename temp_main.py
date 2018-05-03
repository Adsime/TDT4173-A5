from imagehandler import ImageHandler
import extractor as e
from k_nn import KNN
from svm import SVM
import filehandler

def test():
    alphabet = ImageHandler()

    train_images, train_targets = alphabet.get_all_train_data()
    test_images, test_targets = alphabet.get_all_test_data()

    methods = [e.apply_local_threshold]

    train_images = e.extract(methods, train_images, True, True, 40)
    test_images = e.extract(methods, test_images, True)

    knn = KNN()
    svm = SVM()

    knn.train(train_images, train_targets)
    svm.train(train_images, train_targets)

    print(knn.score(test_images, test_targets))
    print(svm.score(test_images, test_targets))


print(filehandler.read_detection_image(1))