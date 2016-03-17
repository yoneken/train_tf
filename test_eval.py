#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2, os, sys, math
import numpy as np
import tensorflow as tf

from mnist import FourLayeredFFNN
from mnist_cnn import FourLayeredFFCNN

class NetEval:
  def __init__(self, num_classes, num_hidden1, num_hidden2):
    self.NUM_CLASSES = num_classes
    self.IMAGE_SIZE = 28
    self.IMAGE_PIXELS = self.IMAGE_SIZE * self.IMAGE_SIZE

    self.images_pl = tf.placeholder(tf.float32, shape=(None, self.IMAGE_PIXELS))
    if FLAGS.one_hot:
      self.labels_pl = tf.placeholder(tf.int32, shape=(None))
    else:
      self.labels_pl = tf.placeholder(tf.int32, shape=(None, self.NUM_CLASSES))

    # Generate network
    self.mnist = FourLayeredFFNN(self.NUM_CLASSES)
    #self.mnist = FourLayeredFFCNN(self.NUM_CLASSES)

    # Initialize network
    self.mnist.init_net(self.images_pl, self.labels_pl, 1e-4)

  def load(self, fname):
    saver = tf.train.Saver()
    saver.restore(self.mnist.sess, fname)

  def eval(self, img):
    img = cv2.resize(img, (28, 28))
    imgg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    imgg = imgg[:,:].reshape(imgg.size)
    imgg = imgg.reshape(1,imgg.size)
    imgf = imgg.astype(np.float32)
    imgf = np.multiply(imgf, 1.0 / 255.0)

    return self.mnist.test(feed_dict={self.images_pl: imgf,
                                     self.labels_pl: [[0] * self.NUM_CLASSES]}).flatten()


if __name__ == '__main__':
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string('img', '', 'image to evaluate')
  flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
  flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
  flags.DEFINE_integer('num_cls', 10, 'Number of classes')
  flags.DEFINE_string('save_file', '', 'Seved session')
  flags.DEFINE_boolean('one_hot', False, 'Use one-hot teaching label.')

  if FLAGS.save_file == '':
    print("You must set a saved session file.")
    exit()

  if FLAGS.img == '':
    print("You must set a path for an image to evaluate.")
    exit()
  img = cv2.imread(FLAGS.img)

  ne = NetEval(FLAGS.num_cls, FLAGS.hidden1, FLAGS.hidden2)
  ne.load(FLAGS.save_file)

  result = ne.eval(img)
  print result
  print np.argmax(result), np.max(result)
