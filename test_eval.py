#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2, os, sys, math
import numpy as np
import tensorflow as tf

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

NUM_CLASSES = 26
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

images_pl = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))
labels_pl = tf.placeholder(tf.int32, shape=(None))

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

logits = inference(images_pl, FLAGS.hidden1, FLAGS.hidden2)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "data-14999")

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

