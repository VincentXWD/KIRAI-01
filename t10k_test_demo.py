import read_mnist
from hpelm import ELM
import numpy as np
from hog_descriptor import *


def get_labels(result):
  ret, id = 0, 0
  for i in range(0, len(result)):
    if ret < result[i]:
      id, ret = i, result[i]
  return id


def main(model_path):
  images, labels = read_mnist.load_mnist('./data', kind='t10k')
  images = map(read_mnist.up_to_2D, images)
  images = map(get_hog, images)
  images = np.mat(np.array(images))

  labels = np.mat(map(read_mnist.handle_label, labels))

  elm = ELM(images.shape[1], labels.shape[1])
  # print images.shape[1], images.shape[1]
  elm.load(model_path)
  results = elm.predict(images)

  labels = map(get_labels, np.array(labels))
  results = map(get_labels, np.array(results))
  yes, tot = 0, len(labels)

  for i in range(0, len(labels)):
    if labels[i] == results[i]:
      yes += 1

  print 'YES :', yes
  print 'TOT :', tot
  print 'ACC : ', str(float(yes)/tot*100.0)+'%'


if __name__ == '__main__':
  main('./elm.model')
