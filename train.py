# -*- coding: utf-8 -*-
import tensorflow as tf
from config import config
import os
import h5py
import numpy as np

checkpoint_path = './checkpoint'
data_dir = './train.h5'


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


def read_train_data(path):
    with h5py.File(path, 'r') as hf:
        input = np.array(hf.get('input'))
        label = np.array(hf.get('label'))
        return input, label


def main(_):  # ?
    with tf.Session() as sess:
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

        def save_checkpoint(step):
            model_dir = "srcnn"
            model_path = os.path.join(checkpoint_path, model_dir)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(sess, os.path.join(model_path, 'model'), global_step=step)

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

        input, label = read_train_data(data_dir)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(labels - conv3))
            tf.summary.scalar('loss', loss)
        saver = tf.train.Saver()
        train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
        tf.initialize_all_variables().run()

        epoch = 0
        counter = 0
        load_checkpoint()

        if not os.path.exists(config.log_path):
            os.mkdir(config.log_path)

        merged = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter("./logs", sess.graph)
        print("Training Start")
        while True:
            batchs_length = len(input) // config.batch_size
            for index in range(0, batchs_length):
                batch_images = input[index * config.batch_size:(index + 1) * config.batch_size]
                batch_labels = label[index * config.batch_size:(index + 1) * config.batch_size]

                _, err = sess.run([train_op, loss], feed_dict={images: batch_images, labels: batch_labels})
                counter += 1

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], loss: [%.8f]" % (
                        (epoch + 1), counter, err))

                if counter % 500 == 0:
                    summary = sess.run(merged, feed_dict={images: batch_images, labels: batch_labels})
                    log_writer.add_summary(summary, counter)
                    save_checkpoint(counter)
            epoch += 1


if __name__ == '__main__':
    tf.app.run()
