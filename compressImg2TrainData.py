#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Directory structure
TRAIN_DIR: 
  label0:
    img0001.png
    img0002.png
    img0003.png
  label1:
    img0001.png
    img0002.png
    .
    .
    .
  label9:
    img0001.png
'''

import cv2, os, gzip, random
import numpy as np
from itertools import chain

class MakeMnistData:
  '''This class makes a train data set and a test data set for MNIST'''

  def __init__(self):
    self.LABEL_MAGIC_NUMBER = 2049
    self.IMAGE_MAGIC_NUMBER = 2051

    self.data_label = []  # the length is the same with the all data
    self.img_data = []    # the length is the same with the all data
    self.data_size = []   # the length is the same with the classes
    self.label_name = []  # the length is the same with the classes

  def _make32(self, val):
    # Big endian
    return [val >> i & 0xff for i in [24,16,8,0]]

  def load(self, dirname):
    for i,dname in enumerate(sorted(next(os.walk(dirname))[1])):
      files = next(os.walk(dirname + "/" + dname))[2]
      self.data_label.append([i]*len(files))
      self.data_size.append(len(files))
      self.label_name.append(dname)

      for filename in files:
        img_file = dirname + "/" + dname + "/" + filename
        img = cv2.imread(img_file)
        img = cv2.resize(img, (28, 28))
        imgg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)

        self.img_data.append(imgg[:,:].reshape(imgg.size))

    self.data_label = list(chain.from_iterable(self.data_label))

  def write(self, dirname, valid_size=0):
    if valid_size == 0:
      valid_size = int(len(self.data_label) * 0.05)

    # make test data
    test_data_label = []
    test_data_size = [0]*len(self.data_label)
    test_img_data = []

    for i in range(valid_size):
      ind = random.randint(0, len(self.data_label)-1)
      test_data_label.append(self.data_label[ind])
      test_img_data.append(self.img_data[ind])

      sind = self.data_label[ind]
      self.data_size[sind] = self.data_size[sind] - 1
      test_data_size[sind] = test_data_size[sind] + 1
      del self.data_label[ind]
      del self.img_data[ind]

    # make a train label data
    # make header
    ldata = self._make32(self.LABEL_MAGIC_NUMBER)
    ldata = np.r_[ldata, self._make32(sum(self.data_size))]
    ldata = np.r_[ldata, self.data_label]

    with gzip.open(dirname + "/train-labels-idx1-ubyte.gz",'wb') as f:
      f.write(np.array(ldata, dtype=np.uint8))

    # make a test label data
    # make header
    tldata = self._make32(self.LABEL_MAGIC_NUMBER)
    tldata = np.r_[tldata, self._make32(sum(test_data_size))]
    tldata = np.r_[tldata, test_data_label]

    with gzip.open(dirname + "/t10k-labels-idx1-ubyte.gz",'wb') as f:
      f.write(np.array(tldata, dtype=np.uint8))

    # make a train image data
    # make header
    idata = self._make32(self.IMAGE_MAGIC_NUMBER)
    idata = np.r_[idata, self._make32(sum(self.data_size))]
    idata = np.r_[idata, self._make32(28)]
    idata = np.r_[idata, self._make32(28)]
    idata = np.r_[idata, list(chain.from_iterable(self.img_data))]

    # write value
    with gzip.open(dirname + "/train-images-idx3-ubyte.gz",'wb') as f:
      f.write(np.array(idata, dtype=np.uint8))

    # make a test image data
    # make header
    tidata = self._make32(self.IMAGE_MAGIC_NUMBER)
    tidata = np.r_[tidata, self._make32(sum(test_data_size))]
    tidata = np.r_[tidata, self._make32(28)]
    tidata = np.r_[tidata, self._make32(28)]
    tidata = np.r_[tidata, list(chain.from_iterable(test_img_data))]

    # write value
    with gzip.open(dirname + "/t10k-images-idx3-ubyte.gz",'wb') as f:
      f.write(np.array(tidata, dtype=np.uint8))

    s = ",".join(["\"" + x + "\"" for x in self.label_name])
    print(s)
    with open(dirname + "/label_name.txt", 'w') as f:
      f.write(s)


if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description="This script makes a train and a validation dataset")
  parser.add_argument("--in_dir", dest="indir", type=str, default="data")
  parser.add_argument("--out_dir", dest="outdir", type=str, default="data")
  parser.add_argument("--valid_size", dest="valsize", type=int, default=0, help="Default size is 5% of all data")
  args = parser.parse_args()
  
  mmd = MakeMnistData()
  mmd.load(args.indir)
  mmd.write(args.outdir, args.valsize)