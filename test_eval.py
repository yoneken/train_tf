#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2, os, sys, math
import numpy as np
import tensorflow as tf

class NetEval:
  def __init__(self, num_classes, num_hidden1, num_hidden2):
    self.NUM_CLASSES = num_classes
    self.IMAGE_SIZE = 28
    self.IMAGE_PIXELS = self.IMAGE_SIZE * self.IMAGE_SIZE

    self.images_pl = tf.placeholder(tf.float32, shape=(None, self.IMAGE_PIXELS))
    self.labels_pl = tf.placeholder(tf.int32, shape=(None))

    self.logits = self.inference(self.images_pl, num_hidden1, num_hidden2)

    self.sess = tf.Session()

  def load(self, fname):
    saver = tf.train.Saver()
    saver.restore(self.sess, fname)

  def inference(self, images, hidden1_units, hidden2_units):
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
          tf.truncated_normal([self.IMAGE_PIXELS, hidden1_units],
                              stddev=1.0 / math.sqrt(float(self.IMAGE_PIXELS))),
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
          tf.truncated_normal([hidden2_units, self.NUM_CLASSES],
                              stddev=1.0 / math.sqrt(float(hidden2_units))),
          name='weights')
      biases = tf.Variable(tf.zeros([self.NUM_CLASSES]),
                           name='biases')
      logits = tf.matmul(hidden2, weights) + biases
    return logits

  def eval(self, img):
    img = cv2.resize(img, (28, 28))
    imgg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    imgg = imgg[:,:].reshape(imgg.size)
    imgg = imgg.reshape(1,imgg.size)
    imgf = imgg.astype(np.float32)
    imgf = np.multiply(imgf, 1.0 / 255.0)

    return self.sess.run(self.logits, feed_dict={self.images_pl: imgf,
                                     self.labels_pl: 0}).flatten()

if __name__ == '__main__':
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string('img', '', 'image to evaluate')
  flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
  flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
  flags.DEFINE_integer('num_cls', 10, 'Number of classes')
  flags.DEFINE_string('save_file', '', 'Seved session')

  if FLAGS.save_file == '':
    print("You must set a saved session file.")
    exit()

  if FLAGS.img == '':
    print("You must set a path for an image to evaluate.")
    exit()
  img = cv2.imread(FLAGS.img)

  ne = NetEval(FLAGS.num_cls, FLAGS.hidden1, FLAGS.hidden2)
  ne.load(FLAGS.save_file)
  
  print np.argmax(ne.eval(img))
