import numpy as np


def add_rotated(images, targets):
    new_imgs = []
    for image in images:
        for i in range(3):
            image = image.rot90()
            new_imgs.append(image)
            targets.append(image.get_target())
    concat(images, new_imgs)


def add_invert(images, targets):
    new_imgs = []
    for image in images:
        new_imgs.append(image.invert())
        targets.append(image.get_target())
    concat(images, new_imgs)


def add_mirror(images, targets):
    new_imgs = []
    for image in images:
        new_imgs.append(image.mirror())
        targets.append(image.get_target())
    concat(images, new_imgs)


def concat(a, b):
    for x in b:
        a.append(x)
