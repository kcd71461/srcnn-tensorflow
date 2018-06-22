# -*- coding: utf-8 -*-
import tensorflow as tf
from config import config
import os
from commons import (read_h5_data, saveImage, mergeSubimages)
from generate_test_h5 import gen

checkpoint_path = './checkpoint'
data_dir = './test.h5'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("test_img", "", "Test img")

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def main(_):
    with tf.Session() as sess:
        num_of_vertical_sub_imgs, num_of_horizontal_sub_imgs = gen(FLAGS.test_img)

        def load_checkpoint():
            print("Load Checkpoint")
            model_dir = "srcnn"
            model_path = os.path.join(checkpoint_path, model_dir)
            checkpoint = tf.train.get_checkpoint_state(model_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                path = str(checkpoint.model_checkpoint_path)
                saver.restore(sess, os.path.join(os.getcwd(), path))
                print("\tLoaded!")
            else:
                print("\tcheckpoint not exists!")


        # <editor-fold desc="placeholder">
        # color channel: 3
        images = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, 3], name='images')
        labels = tf.placeholder(tf.float32, [None, config.label_size, config.label_size, 3], name='labels')
        # </editor-fold>

        # <editor-fold desc="weight & bias">
        weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=0.001), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=0.001), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=0.001), name='w3')
        }

        biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([3], name='b3'))
        }
        # </editor-fold>

        # <editor-fold desc="convolutional Layer">
        conv1 = tf.nn.relu(
            tf.nn.conv2d(images, weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b1'])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])
        conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']
        # </editor-fold>

        input, label = read_h5_data(data_dir)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(labels - conv3))
            tf.summary.scalar('loss', loss)
        saver = tf.train.Saver()

        load_checkpoint()

        result = conv3.eval({images: input})
        image = mergeSubimages(result, [num_of_vertical_sub_imgs, num_of_horizontal_sub_imgs])
        if not os.path.exists(config.result_dir):
            os.mkdir(config.result_dir)
        saveImage(image, config.result_dir + '/result.png')


if __name__ == '__main__':
    tf.app.run()
