import os
import numpy as np
import librosa
from params import *


def extractFeature(load_path, save_path):
	# check dataset
	assert os.path.exists(load_path) == True
	if not os.path.exists(save_path): os.mkdir(save_path)
	data_list = os.listdir(load_path)

	# extract spectra
	for i, data in enumerate(data_list):
	    print('[%4d / %4d]'%(i+1, len(data_list)), data.replace('.wav', ''))
	    x, _    = librosa.load(load_path+'/'+data, sr, mono=False)
	    n_frame  = int(x.shape[1] / hop_length) + 1
	    spectra = np.zeros((n_frame, f_bins, n_src), dtype=np.complex64)
	    for i in range(n_src):
	        spectra[..., i] = (librosa.stft(y=x[i, :], n_fft=n_fft, hop_length=hop_length)).T
	    np.save(file=save_path+'/'+data.replace('.wav', '.npy'), arr=spectra)


if __name__ == '__main__':
	extractFeature('./dataset/Wavfile', './dataset/spectra')