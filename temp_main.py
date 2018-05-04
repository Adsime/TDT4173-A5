from imagehandler import ImageHandler
import extractor as e
from k_nn import KNN
from svm import SVM
from ann import ANN
from copy import deepcopy
import augmenter

def test():
    alphabet = ImageHandler(0.20)

    train_images, train_targets = alphabet.get_all_train_data()
    test_images, test_targets = alphabet.get_all_test_data()

    #augmenter.add_rotated(train_images, train_targets)
    #augmenter.add_mirror(train_images, train_targets)
    augmenter.add_invert(train_images, train_targets)


    methods = [[e.perform_edge_detection, e.calculate_hog], [e.calculate_hog], [e.apply_otsu_threshold], [e.apply_local_threshold], [e.apply_yen_threshold],
               [e.apply_mean_threshold], [e.apply_li_threshold], [e.apply_isodata_threshold],
               [e.apply_triangle_threshold]]

    methods = [[e.apply_otsu_threshold], [e.apply_local_threshold], [e.apply_yen_threshold],
               [e.apply_mean_threshold], [e.apply_li_threshold], [e.apply_isodata_threshold],
               [e.apply_triangle_threshold]]

    methods = [[e.apply_local_threshold]]

    #methods = [[e.apply_gaussian_filter]]

    for method in methods:
        svm = ANN()
        tri = e.extract(deepcopy(method), deepcopy(train_images), True, True, 30)
        tei = e.extract(deepcopy(method), deepcopy(test_images), True, False)
        #tri = e.extract(deepcopy(method), deepcopy(train_images), False, False, 30)
        #tei = e.extract(deepcopy(method), deepcopy(test_images), False, False)

        svm.train(tri, train_targets)
        print("Method: " + method[0].__name__ + " - Error: " + svm.score(tei, test_targets).__str__())

    """knn = KNN()
    svm = SVM()

    knn.train(train_images, train_targets)
    svm.train(train_images, train_targets)

    print(knn.score(test_images, test_targets))
    print(svm.score(test_images, test_targets))
    """
test()