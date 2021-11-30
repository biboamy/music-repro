import os
import numpy as np
import tqdm
import glob
import torch
import librosa

class Processor:
	def __init__(self):
		self.fs = 16000
		self.vgg_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
		self.vgg_model.eval()

	def get_paths(self, data_path):
		#self.files = glob.glob(os.path.join(data_path, 'mtat', 'mp3', '*/*.mp3'))
		self.files = glob.glob(os.path.join(data_path, 'GTZAN', 'genres', '*/*.wav'))
		self.npy_path = os.path.join(data_path, 'gtzan_vgg', 'npy')
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)

	def get_npy(self, fn):
		y, sr = librosa.load(fn)
		x = self.vgg_model.forward(y, sr).detach().cpu().numpy()
		return x

	def iterate(self, data_path):
		self.get_paths(data_path)
		for fn in tqdm.tqdm(self.files):
			npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy')
			if not os.path.exists(npy_fn):
				try:
					
					x = self.get_npy(fn)
					np.save(open(npy_fn, 'wb'), x)
				except:
					# some audio files are broken
					print(fn)
					continue

if __name__ == '__main__':
	p = Processor()
	p.iterate('/home/yhung/data/gtzan/')
