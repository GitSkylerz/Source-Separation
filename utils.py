import os
import numpy as np
from random import shuffle
from librosa.output import write_wav
from librosa import stft, istft


def writeWav(path, X):
    x = istft(X, 256, 1024)
    x = x - np.mean(x)
    write_wav(path, x, 16000, True)

    
class Generator(object):
    
	def __init__(self, data_list, batch_size, t_bins, f_bins, n_mic, n_src,
				 strids=1, is_rand=False):
		assert data_list is not None
	    self.data_list  = data_list
	    self.data_list.sort()
	    self.batch_size = batch_size
	    self.t_bins     = t_bins
	    self.f_bins     = f_bins
	    self.n_mic      = n_mic
	    self.n_src      = n_src
	    self.is_rand    = is_rand

	    # load data
	    print('Loading data ...')
	    self._data_pool = {i:np.load(i) for i in self.data_list}
	    print('Finish loading data !')

	    # random indexes
	    for i, data in self._data_pool.items():
	        self._access_idx = [(i, j) for j in range(self.t_bins, len(data)+1)]
	    self.on_epoch_end()
	    
	    
	def __len__(self):
	    # Method to calculate total number of batch in one epoch
	    n_batch = len(self._access_idx) // self.batch_size
	    if len(self._access_idx) % self.batch_size:
	        n_batch += 1
	        
	    return n_batch
	    
	    
	def __getitem__(self, i):
	    # Method to generate one batch of data
	    sub_idx = self._access_idx[i*self.batch_size : (i+1)*self.batch_size]

	    return self.__data_generation(sub_idx)
	    
	    
	def __iter__(self):
	    # Create a iterator that iterate over the sequence
	    for i in range(len(self)):
	        yield self[i]
	        
	        
	def on_epoch_end(self):
	    # Updates indexes after each epoch
	    if self.is_rand: 
	        shuffle(self._access_idx)
	        
	        
	def __data_generation(self, sub_idx):
	    # Generates data containing batch_size samples
	    features = np.zeros([len(sub_idx), self.t_bins, self.f_bins, 2*self.n_mic],
	    					dtype=np.float32)
	    targets  = np.zeros([len(sub_idx), self.t_bins, self.f_bins, 2*self.n_src],
	    					dtype=np.float32)
	    for i, idx in enumerate(sub_idx):
	    	# indexing
	        frame_mask    = np.arange(idx[1]-self.t_bins, idx[1])
	        clean_spectra = self._data_pool[idx[0]][frame_mask, ...]
	        mixed_spectra = np.mean(clean_spectra, axis=-1)
	            
	        # features
	        for j in range(self.n_mic):
	        	features[i, ..., j] = np.abs(mixed_spectra)
	        	features[i, ..., j+self.n_mic] = np.angle(mixed_spectra)
	        
	        # targets
	        for j in range(self.n_src):
	            targets[i, ..., j] = np.abs(clean_spectra[..., j])
	            targets[i, ..., j+self.n_src] = np.angle(clean_spectra[..., j])
	            
	    return features, targets
        
        
class Solver(object):
    
    def __init__(self, model=None, generator=None):
        self.model     = model
        self.generator = generator
            
            
    def train(self, epoch, is_evaluate=False, early_stop=False):
        n_batch = len(self.generator)
        for i in range(epoch):
            loop_counter = 0
            for x, y in self.generator:
                loop_counter += 1
                self.model._fit(x=x, y=y)
                _, train_loss = self.model._eval(x=x, y=y)
                print('#%3d:  [%4d/ %4d] Training Loss = %5.4f'%(i+1, loop_counter,
                                                                 n_batch, train_loss), end='\r') 

            print('#%3d:  [%4d/ %4d] Training Loss = %5.4f'%(i+1, loop_counter,
                                                             n_batch, train_loss), end='\n')  
            self.generator.on_epoch_end()
    
    
    def inference(self, x, t_bins, f_bins, n_mic, n_src):
        # load time domain audio data
        assert x.shape[0] is n_mic
        n_frame = int(x.shape[1] / hop_length) + 1
        mixed_spectra = np.zeros((n_frame, f_bins, n_mic), dtype=np.complex64)
        for i in range(n_mic):
        	mixed_spectra[..., i] = (librosa.stft(y=x[i, :], n_fft=n_fft, hop_length=hop_length)).T

        # load data and zero-padding for batch calculation
        n_batch = n_frame // t_bins
        if n_frame % t_bins:
            mixed_spectra = np.concatenate((mixed_spectra, 
            	np.zeros((t_bins-n_frame%t_bins, f_bins, n_mic))), axis=0)
            n_batch += 1
        
        # input features
        mixed_spectra_magnitude = np.abs(mixed_spectra).reshape((n_batch, t_bins, f_bins, n_mic))
        mixed_spectra_phase     = np.angle(mixed_spectra).reshape((n_batch, t_bins, f_bins, n_mic))
        input_features          = np.concatenate((mixed_spectra_magnitude, 
        										  mixed_spectra_phase), axis=-1)

        # output features
        output_features   = self.model._pred(x=input_features)
        estimated_spectra = (output_features[..., :n_src] + \
        					 1j * output_features[..., n_src:2*n_src]) * \
        					np.exp(1j * mixed_spectra_phase[..., [0]])
        estimated_spectra = estimated_spectra.reshape((n_batch*t_bins, f_bins, n_src))
        if n_frame % t_bins:
            estimated_spectra = estimated_spectra[:n_frame, ...]

        for i in range(n_src):
        	x = istft(estimated_spectra[..., i].T, hop_length, n_fft)
        	x = x - np.mean(x)
        	write_wav(path, x, sr, True)
        return estimated_features


    def saveModel(self, path):
        self.model._save(path=path)
    
    
    def restoreModel(self, path, keyword=None):
        self.model._restore(path=path, keyword=keyword)


if __name__ == '__main__':
	# Testing for Generator
	from params import *
	path = './dataset/spectra'
	data_list = [path+'/'+i for i in os.listdir(path)]
	g = Generator(data_list, 8, f_bins, n_mic, n_src, batch_size, is_rand=False)
	for features, targets in g:
		print(features.shape)
		print(targets.shape)