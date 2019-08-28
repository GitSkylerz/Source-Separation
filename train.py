import os
import numpy as np
import tensorflow
from model import *
from utils import *
import sys
from params import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(dataset_path):
    # load dataset
    data_list = [dataset_path+'/'+i for i in os.listdir(dataset_path)]
    print('# Total data:', len(data_list))

    # initialize generator and model
    train_generator = Generator(data_list, batch_size, t_bins, f_bins, n_mic, n_src, True)
    model           = MaskCRNN(t_bins, f_bins, n_mic, n_src)
    solver          = Solver(model, train_generator)

    # training processure
    solver.train(epoch, True, early_stop)

    # save model parameters
    solver.saveModel(path='./model_params/MaskCRNN.ckpt')


if __name__ == '__main__':
    train('./dataset/spectra')