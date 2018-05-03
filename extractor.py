import pca as p
from skimage import filters, feature
import skimage.filters as f
from image import Image
import numpy as np

pca = None


def extract(methods, images, use_pca=False, fit=False, n_components=30):
    global pca
    methods.append(flatten)
    for method in methods:
        for i, img in enumerate(images):
            images[i] = method(img.get_image()) if (type(img) is Image) else method(img)
    if use_pca:
        if fit:
            pca = p.get_pca(n_components, images)
            pca.fit(images)
        return pca.transform(images)
    return images


def flatten(image):
    return image.flatten()


def apply_gaussian_filter(image):
    return filters.gaussian(image, 1)


def perform_edge_detection(image):
    return feature.canny(apply_gaussian_filter(image))


def apply_otsu_threshold(image):
    return image > f.threshold_otsu(image)


def apply_local_threshold(image):
    return image > f.threshold_local(image,50)


def calculate_hog(image):
    return feature.hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))


def calculate_lbp(image):
    return feature.local_binary_pattern(image, 8, 1)


def calculate_hog_and_lbp(image):
    return np.concatenate((calculate_hog(image), calculate_lbp(image).flatten()))