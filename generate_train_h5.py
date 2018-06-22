# -*- coding: utf-8 -*-
import cv2
from config import config
import os
import glob
import numpy as np
import h5py


def load_image_paths():
    """
datasets 내 bitmap file path들 반환
    :return:
    """
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    return glob.glob(os.path.join(datasets_path, "*.bmp"))


def crop_for_label_image(image, scale):
    """
scale down, up 한 후 이미지 크기에 맞게 crop
    :param image: image
    :param scale: scale 예정인 크기
    :return: image
    """
    height, width, _ = image.shape
    height = int((height / scale) * scale)
    width = int((width / scale) * scale)
    return image[0:int(height), 0:int(width), :]


def generateInputAndLabelImage(image_path, scale):
    """
input, label(opencv bicubic scaling을 이용해서 scale down & scale up 한 이미지 생성) 생성
    :param image_path:
    :param scale:
    :return:
    """
    image = cv2.imread(image_path)
    label_ = crop_for_label_image(image, scale)
    bicbuic_img = cv2.resize(label_, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)  # scale down
    input_ = cv2.resize(bicbuic_img, None, fx=scale, fy=scale,  # scale up
                        interpolation=cv2.INTER_CUBIC)  # Resize by scaling factor
    return input_, label_


def save_h5_data(input, label):
    with h5py.File(os.path.join(os.getcwd(), 'train.h5'), 'w') as hf:
        hf.create_dataset('input', data=input)
        hf.create_dataset('label', data=label)


def generate_sub_input_label(imagePaths, padding):
    """
sub input-label들 생성
    :param imagePaths:
    :param padding:
    :return:
    """
    sub_inputs_ = []
    sub_labels_ = []
    for i in range(len(imagePaths)):
        input, label, = generateInputAndLabelImage(imagePaths[i], config.scale)
        height, weight, color = input.shape

        # <editor-fold desc="sub input-label 생성">
        for x in range(0, height - config.image_size + 1, config.stride):
            for y in range(0, weight - config.image_size + 1, config.stride):
                sub_input = input[x: x + config.image_size, y: y + config.image_size]
                sub_label = label[x + padding: x + padding + config.label_size,
                            y + padding: y + padding + config.label_size]

                # 3개 채널로 reshape
                sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

                # 0 ~ 255 to 0.0 ~ 1.0
                sub_input = sub_input / 255.0
                sub_label = sub_label / 255.0

                sub_inputs_.append(sub_input)
                sub_labels_.append(sub_label)
        # </editor-fold>

    return sub_inputs_, sub_labels_


imagePaths = load_image_paths()
padding = int(abs(config.image_size - config.label_size) / 2)

# sub_inputs,sub_labels 생성
sub_inputs, sub_labels = generate_sub_input_label(imagePaths, padding)

# numpy array로 변환
inputArr = np.asarray(sub_inputs)
inputLabel = np.asarray(sub_labels)

save_h5_data(inputArr, inputLabel)
