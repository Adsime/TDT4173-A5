import pca as p
from skimage import filters, feature
import skimage.filters as f

pca = None


def extract(methods, images, use_pca=False, fit=False, n_components=30):
    global pca
    for method in methods:
        for i, img in enumerate(images):
            images[i] = method(img.get_image())
    if use_pca:
        if fit:
            pca = p.get_pca(n_components, images)
            pca.fit(images)
        return pca.transform(images)
    return images


def apply_gaussian_filter(image, flatten=True):
    ret_val = filters.gaussian(image, 1)
    return ret_val.flatten() if flatten else ret_val


def perform_edge_detection(image, flatten=True):
    ret_val = feature.canny(apply_gaussian_filter(image, False))
    return ret_val.flatten() if flatten else ret_val


def apply_otsu_threshold(image, flatten=True):
    ret_val = f.threshold_otsu(image)
    return (image > ret_val).flatten() if flatten else image > ret_val