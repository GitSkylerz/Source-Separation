import os
import sys
import numpy as np
import tensorflow
from model import *
from utils import *
import librosa
import sklearn
import params
from params import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def inference(dataset_path):
    # load dataset
    data_list = [dataset_path+'/'+i for i in os.listdir(dataset_path)]
    print('# Total data:', len(data_list))

    # initialize generator and model
    test_generator = Generator(data_list, batch_size, t_bins, f_bins, n_mic, n_src, True)
    model          = MaskCRNN(t_bins, f_bins, n_mic, n_src)
    solver         = Solver(model, test_generator)

    # restore example
    solver.restoreModel(path='./model_params')

    # inference
    y_true = []
    y_pred = []
    for k, data in enumerate(eval_data_list):
        print('[%d / %d]'%(k+1, len(eval_data_list)), end='\r')
        # seperation
        file_name = data[0].split('/')[-1].replace('.wav', '').replace('.npy', '')
        separated_feature, _, _ = solver.inference(data[0], time_step, nb_mic, nb_trg, nb_t, nb_f)
        for i in range(nb_trg):
            x = istft(separated_feature[..., i].T, 256, 1024)
            x = x / np.max(np.abs(x))
            librosa.output.write_wav('./outputs'+dataset_name+acoustic_env+'/'+file_name+'_src'+str(i)+'.wav', x, 16000)
            

if __name__ == '__main__':
    inference(dataset_name, num_part)