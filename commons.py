import os
import glob
import h5py
import numpy as np
import cv2
import tensorflow as tf
from config import config


def load_image_paths(path):
    """
datasets 내 bitmap file path들 반환
    :return:
    """
    datasets_path = os.path.join(os.getcwd(), path)
    return glob.glob(os.path.join(datasets_path, "*.bmp"))


def save_h5_data(input, label, filename):
    with h5py.File(os.path.join(os.getcwd(), filename), 'w') as hf:
        hf.create_dataset('input', data=input)
        hf.create_dataset('label', data=label)


def read_h5_data(path):
    with h5py.File(path, 'r') as hf:
        input = np.array(hf.get('input'))
        label = np.array(hf.get('label'))
        return input, label


def saveImage(image, filename):
    cv2.imwrite(os.path.join(os.getcwd(), filename), image * 255.)


def mergeSubimages(images, size):
    height, width = images.shape[1], images.shape[2]

    img = np.zeros((height * size[0], width * size[1], 3))
    for idx, image in enumerate(images):
        horizontalIndex = idx % size[1]
        verticalIndex = idx // size[1]
        img[verticalIndex * height: verticalIndex * height + height,
        horizontalIndex * width: horizontalIndex * width + width, :] = image

    return img


def scaleDownAndUp(image, scale):
    bicbuic_img = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)  # scale down
    return cv2.resize(bicbuic_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # scale up


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(img1, img2):
    img_arr1 = np.array(img1).astype('float32')
    img_arr2 = np.array(img2).astype('float32')
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255 ** 2, dtype=tf.float32) / mse
    result = tf.constant(10, dtype=tf.float32) * log10(psnr)
    return result.eval()