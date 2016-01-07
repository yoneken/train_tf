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

import cv2, os
import numpy as np

LABEL_MAGIC_NUMBER = 2049
IMAGE_MAGIC_NUMBER = 2051
TRAIN_DIR = "data"

def _make32(val):
  # Big endian
  return [val >> i & 0xff for i in [24,16,8,0]]

img_data = []
data_label = []
data_size = {}

for dirname in next(os.walk(TRAIN_DIR))[1]:
  data_label.append(dirname)

  files = next(os.walk(TRAIN_DIR + "/" + dirname))[2]
  data_size[dirname] = len(files)

  for filename in files:
    img_file = TRAIN_DIR + "/" + dirname + "/" + filename
    #print(img_file)
    img = cv2.imread(img_file)
    img = cv2.resize(img, (28, 28))
    imgg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)

    img_data = np.r_[img_data, imgg[:,:].reshape(imgg.size)]


# make a train label data
# make header
ldata = _make32(LABEL_MAGIC_NUMBER)
ldata = np.r_[ldata, _make32(sum(data_size.values()))]

# write value
for i,v in enumerate(data_label):
  ldata = np.r_[ldata, [i]*data_size[v]]

np.array(ldata, dtype=np.uint8).tofile(TRAIN_DIR + "/labels.idx1-ubyte")

# make a train image data
# make header
idata = _make32(IMAGE_MAGIC_NUMBER)
idata = np.r_[idata, _make32(sum(data_size.values()))]
idata = np.r_[idata, _make32(28)]
idata = np.r_[idata, _make32(28)]
idata = np.r_[idata, img_data]

np.array(idata, dtype=np.uint8).tofile(TRAIN_DIR + "/images.idx3-ubyte")

with open(TRAIN_DIR + "/label_name.txt", 'w') as f:
  f.write(",".join(["\"" + x + "\"" for x in data_label]))

