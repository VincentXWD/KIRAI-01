import model_training
import t10k_test_demo
import datetime
import hog_descriptor

nbins = [8, 9, 18, 24, 32, 64, 128]

def experiment():
  result = []
  for n in nbins:
    time = 3
    tot = 0.0
    print 'now nbins is ', n
    while time > 0:
      hog_descriptor.nbins = n
      begin = datetime.datetime.now()
      model_training.training(model_path='../models/elm.model',
                              data_path='../data/')
      end = datetime.datetime.now()
      tot += t10k_test_demo.t10k_test()
      time -= 1
    result.append((n, tot / 3.0, (end-begin).seconds))
  return result


def write_file(content, fpath, fname='nbins_ctrl_HOG'):
  with open(fpath+fname, 'w') as fp:
    for line in content:
      fp.writelines(str(line)+'\n')


if __name__ == '__main__':
  result = experiment()
  write_file(result, '../')
