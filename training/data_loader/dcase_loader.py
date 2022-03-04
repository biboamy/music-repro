from torch.utils import data
import json
import numpy as np
import torch
import soundfile as sf
import os
import random
import pandas as pd

class DCASE17(data.Dataset):
	def __init__(self, root, split, input_length=None, model='resnet18'):
		split = split.lower()

		df = pd.read_csv(os.path.join(root, 'df.csv'), delimiter='\t', names=['file', 'start', 'end', 'path', 'split', 'label'])

		if split == 'train':
			df = df[df['split']=='train']
		elif split == 'test':
			df = df[df['split']=='test']
		else:
			df = df[df['split']=='val']
		
		self.files = list(df['path'])
		self.binary = list(df['label'])
		print(len(self.files))

		self.split = split
		self.seg_length = input_length
		self.root = root
		self.model = model
		if self.model == 'hubert_ks':
			self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")


	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		
		if self.split == 'train':
			fn = self.files[idx].replace('~', '../..')
			audio = np.load(fn, mmap_mode='r')

			if len(audio) < self.seg_length:
				nnpy = np.zeros(self.seg_length)
				nnpy[0:0+len(audio)] = audio
				audio = nnpy

			random_idx = int(np.floor(np.random.random(1) * (len(audio)-self.seg_length)))
			audio = np.array(audio[random_idx:random_idx+self.seg_length])

			label = np.fromstring(self.binary[idx][1:-1], dtype=np.float32, sep=' ')
			
			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
			return audio.astype('float32'), label.astype('float32')
			
		else:
			fn = self.files[idx].replace('~', '../..')
			audio = np.load(fn, mmap_mode='r').astype('float32')
			label = np.fromstring(self.binary[idx][1:-1], dtype=np.float32, sep=' ')

			if len(audio) < self.seg_length:
				nnpy = np.zeros(self.seg_length)
				nnpy[0:0+len(audio)] = audio
				audio = nnpy
			
			n_chunk = len(audio) // self.seg_length 
			
			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
					audio_chunks = torch.stack(torch.split(audio[d][:int(n_chunk*self.seg_length)], n_chunk))
					torch.cat((audio_chunks, audio[d][-int(self.seg_length):][:, None]), -1)
					audio[d] =  audio_chunks.T
				
			elif 'resnet' in self.model or 'CNN' in self.model or 'speechatt' in self.model:
				audio_chunks = np.split(audio[:int(n_chunk*self.seg_length)], n_chunk)
				audio_chunks.append(audio[-int(self.seg_length):])
				audio = np.array(audio_chunks)

			return audio.astype('float32'), label.astype('float32')


def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None, model='resnet18'):
	if split == 'train':
		shuffle = True
	else:
		shuffle = False

	data_loader = data.DataLoader(dataset=DCASE17(root, split=split, input_length=input_length, model=model),
								  batch_size=batch_size,
								  shuffle=shuffle,
								  drop_last=False,
								  num_workers=num_workers)
	return data_loader

