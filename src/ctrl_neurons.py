import datetime

import model_training
from demos import t10k_test_demo


def experiment(lo, hi, step=50):
  result = []
  for neurons in range(lo, hi, step):
    time = 3
    tot = 0.0
    while time > 0:
      print neurons
      begin = datetime.datetime.now()
      model_training.training(model_path='../models/elm.model',
                              data_path='../data/',
                              neurons=neurons)
      end = datetime.datetime.now()
      tot += t10k_test_demo.t10k_test()
      time -= 1
    result.append((neurons, tot / 3.0, end-begin))
  return result


def write_file(content, fpath, fname='neurons_ctrl'):
  with open(fpath+fname, 'w') as fp:
    for line in content:
      fp.writelines(str(line)+'\n')


if __name__ == '__main__':
  result = experiment(900, 1001, 100)
  write_file(result, '../')
