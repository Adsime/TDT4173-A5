from alphabet import Alphabet
import matplotlib.pyplot as plt

alphabet = Alphabet()


imgs = alphabet.get_images('b')
img = imgs[114]
plt.imshow(img.get_image_map(), cmap="gray")
plt.figure()


plt.imshow(img.smooth(), cmap="gray")
plt.show()