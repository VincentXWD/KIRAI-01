# -*- coding:utf-8 -*-
from hpelm import ELM
import read_mnist
import numpy as np
from hog_descriptor import *



def model_training(model_path):
  images, labels = read_mnist.load_mnist('./data', kind='train')

  images = map(read_mnist.up_to_2D, images)
  images = map(get_hog, images)
  images = np.mat(np.array(images))

  labels = np.mat(map(read_mnist.handle_label, labels))

  elm = ELM(images.shape[1], labels.shape[1])
  elm.add_neurons(300, 'tanh')
  elm.add_neurons(300, 'rbf_l2')
  elm.train(images, labels)
  elm.save(model_path)


model_training('./elm.model')
