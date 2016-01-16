#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2, os, sys
import numpy as np
import tensorflow as tf

import mnist

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('img', '', 'image to evaluate')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

if FLAGS.img == '':
  print("You must set a path for an image to evaluate.")
  exit()

images_pl = tf.placeholder(tf.float32, shape=(None, mnist.IMAGE_PIXELS))
labels_pl = tf.placeholder(tf.int32, shape=(None))

logits = mnist.inference(images_pl, FLAGS.hidden1, FLAGS.hidden2)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "data-15999")

img = cv2.imread(FLAGS.img)
img = cv2.resize(img, (28, 28))
imgg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
imgg = imgg[:,:].reshape(imgg.size)
imgg = imgg.reshape(1,imgg.size)
imgf = imgg.astype(np.float32)
imgf = np.multiply(imgf, 1.0 / 255.0)


result = sess.run(logits, feed_dict={images_pl: imgf,
                                     labels_pl: 0}).flatten()

print np.argmax(result)

