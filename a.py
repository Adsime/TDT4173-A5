from image_handler import Alphabet
from k_nn import KNN
from svm import SVM
from matplotlib import pyplot as plt

alphabet = Alphabet()

k_nn = KNN()
svm = SVM()

training_data = alphabet.get_all_train_data()
t_data = []
letters = []
for i, letter in enumerate(training_data):
    for image in letter:
        t_data.append(image.get_filtered_image().flatten())
        letters.append(i)

k_nn.train(t_data, letters)
svm.train(t_data, letters)

test_data = alphabet.get_all_test_data()

knn_error = 0
svm_error = 0
elements = 0
for i, letter in enumerate(test_data):
    for image in letter:
        knn_error += 0 if k_nn.predict(image.get_filtered_image().flatten().reshape(1, -1)) - i == 0 else 1
        svm_error += 0 if svm.predict(image.get_filtered_image().flatten().reshape(1, -1)) - i == 0 else 1
        elements += 1
knn_error /= elements
svm_error /= elements

print("Percentage of letters correctly classified (KNN):", 1-knn_error)
print("Percentage of letters correctly classified (SVM):", 1-svm_error)

for i in range(0,26):
    print("\nExpected: ", i)
    print("KNN predicted: ", k_nn.predict(test_data[i][0].get_filtered_image().flatten().reshape(1, -1)))
    print("SVM predicted: ", svm.predict(test_data[i][0].get_filtered_image().flatten().reshape(1, -1)))
    plt.imshow(test_data[i][0].get_filtered_image(), cmap='gray')
    plt.show()
