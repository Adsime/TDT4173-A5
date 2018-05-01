from alphabet import Alphabet
import matplotlib.pyplot as plt

alphabet = Alphabet()

print(len(alphabet.get_test_set('a')))
print(len(alphabet.get_train_set('a')))
imgs = alphabet.get_images('b')
img = imgs[114]
plt.imshow(img.get_image_map(), cmap="gray")
plt.figure()


plt.imshow(img.smooth(), cmap="gray")
plt.show()