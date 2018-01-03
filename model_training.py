# -*- coding:utf-8 -*-
from PIL import Image
from hpelm import ELM
import read_mnist
import cv2
import numpy as np


def up_to_2D(vec):
  tmp = np.mat(vec).reshape(28, 28)
  img = Image.fromarray(tmp.astype(np.uint8))
  img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
  return img


def do_filter(img):
  pass


def model_training():
  images, labels = read_mnist.load_mnist('./data')
  images = map(up_to_2D, images)
  # cnt = 0
  # for vec in vecs:
  #   print labels[cnt]
  #   cnt += 1
  #   cv2.imshow('P', np.mat(vec))
  #   cv2.waitKey()
  vec = map(do_filter, images)

#   elm = ELM(vecs.shape[1], labels.shape[1])
#   elm.add_neurons(20, "sigm")
#   elm.add_neurons(10, "rbf_l2")
#   elm.train(vecs, labels)


model_training()

