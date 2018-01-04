import os
import struct
import numpy as np
from PIL import Image
import cv2

def load_mnist(path, kind='train'):
  """Load MNIST data from `path`"""
  labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
  images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
  with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

  with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    # cv2.imshow('',np.mat(images).resize(0,0))
    # cv2.waitKey()
  return images, labels


def handle_label(label):
  ret = [0 for _ in range(10)]
  ret[label] = 1.0
  return ret


def up_to_2D(vec):
  tmp = np.mat(vec).reshape(28, 28)
  img = Image.fromarray(tmp.astype(np.uint8))
  _, img = cv2.threshold(np.mat(img), 80, 255, cv2.THRESH_BINARY)
  return img

if __name__ == '__main__':
  load_mnist('./data/')