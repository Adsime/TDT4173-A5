from image_handler import Alphabet
from k_nn import KNN
from svm import SVM
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import image
import pca as PCA

alphabet = Alphabet(0.20)

k_nn = KNN()
svm = SVM()

methods = [image.get_threshhold_image_flat]

n = 10

for i in range(25, 35):
    error = 0
    ind_errs = []
    for j in range(n):
        train_data, train_targets = alphabet.get_all_train_data(methods)
        test_data, test_targets = alphabet.get_all_test_data(methods)
        pca = PCA.get_pca(i, train_data)
        svm = KNN()
        svm.train(PCA.transform(pca, train_data), train_targets)
        elements = 0
        err = 0
        tr_test_data = PCA.transform(pca, test_data)
        print(svm.predict(tr_test_data))
        for indx, img in enumerate(tr_test_data):
            err += 0 if svm.predict(img.reshape(1, -1)) == test_targets[indx] else 1
            elements += 1
        error += err/elements
        ind_errs.append(err/elements)
    print("svm for n_components = " + i.__str__() + " and n = " + n.__str__() + " - Error: " + (error/n).__str__())
    print(ind_errs.__str__())
    print()



"""svm = SVM()

training_data = alphabet.get_all_train_data()
pca = PCA(n_components=30)

t_data = []
letters = []
for i, letter in enumerate(training_data):
    for img in letter:
        t_data.append(image.get_threshhold_image_flat(img))
        letters.append(i)

t_data = pca.fit_transform(t_data)

k_nn.train(t_data, letters)
svm.train(t_data, letters)

test_data = alphabet.get_all_test_data()

knn_error = 0
svm_error = 0
elements = 0

for i, letter in enumerate(test_data):
    for img in letter:
        #get_filtered_image().flatten().reshape(1, -1)
        knn_error += 0 if k_nn.predict(pca.transform(image.get_threshhold_image_flat(img).reshape(1, -1))) - i == 0 else 1
        svm_error += 0 if svm.predict(pca.transform(image.get_threshhold_image_flat(img).reshape(1, -1))) - i == 0 else 1
        elements += 1
knn_error /= elements
svm_error /= elements

print("Percentage of letters correctly classified (KNN):", 1-knn_error)
print("Percentage of letters correctly classified (SVM):", 1-svm_error)

for i in range(0,26):
    print("\nExpected: ", i)
    print("KNN predicted: ", k_nn.predict(pca.transform(image.get_threshhold_image_flat(test_data[i][0]).reshape(1, -1))))
    print("SVM predicted: ", svm.predict(pca.transform(image.get_threshhold_image_flat(test_data[i][0]).reshape(1, -1))))
    plt.imshow(image.get_threshhold_image(test_data[i][0]), cmap='gray')
    plt.show()"""
