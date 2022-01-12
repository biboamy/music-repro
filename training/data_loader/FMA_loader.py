import os
import numpy as np
from torch.utils import data
import random
import soundfile as sf
import sys 
import tqdm
from pandas import *
import librosa


class FMA(data.Dataset):
	def __init__(self, root, split, input_length=None, model='resnet18'):
		split = split.lower()
		query = read_csv(f'{root}/tracks.csv', header=1)
		if split == 'train':
			df = query.loc[((query['subset'] == 'medium') | (query['subset'] == 'small')) & (query['split'] == 'training')]
		elif split == 'test':
			df = query.loc[((query['subset'] == 'medium') | (query['subset'] == 'small')) & (query['split'] == 'test')] # test
		else:
			df = query.loc[((query['subset'] == 'medium') | (query['subset'] == 'small')) & (query['split'] == 'validation')] # test

		self.mappeing = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Easy Listening': 3, 'Electronic': 4, 'Experimental': 5, 'Folk': 6, 'Hip-Hop': 7, \
						 'Instrumental': 8, 'International': 9, 'Jazz': 10, 'Old-Time / Historic': 11, 'Pop': 12, 'Rock': 13, 'Soul-RnB': 14, 'Spoken': 15}
		self.files = df[df.columns[0]].tolist()
		print(len(self.files))

		#if split == 'train':
		#	self.files = random.sample(self.files, int(len(self.files)*0.1))
		self.class_num = 16
		self.split = split
		self.seg_length = input_length
		self.root = root
		self.model = model
		self.genres = df['genre_top'].tolist()
		if self.model == 'hubert_ks':
			self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")


	def __len__(self):
		if self.split == 'train':
			return 10000
		else:
			return len(self.files)

	def __getitem__(self, idx):
		file = f'{int(self.files[idx]):06d}'
		
		label = np.zeros(self.class_num)
		label[self.mappeing[self.genres[idx]]] = 1
		if self.split == 'train':
			try:
				frame = sf.info(os.path.join(self.root, 'audio_16000', file+'.wav')).frames
				start = random.randint(0, frame-self.seg_length-16000)
				end = start + self.seg_length
				audio, sr = sf.read(os.path.join(self.root, 'audio_16000', file+'.wav'), start=start, stop=end)
			except:
				return self.__getitem__(random.randint(0, len(self.files)-1))
			audio = audio.astype('float32')
			
			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
			if len(audio.shape) == 2:
				audio = (audio[:, 0]+audio[:, 1])/2

			return audio, label.astype('float32')
			
		else:
			try:
				audio, sr = sf.read(os.path.join(self.root, 'audio_16000', file+'.wav'))
			except:
				#audio, sr = librosa.load(os.path.join(self.root, 'audio', file+'.mp3'), sr=16000)
				return self.__getitem__(random.randint(0, len(self.files)-1))

			audio = audio.astype('float32')

			if len(audio.shape) == 2:
				audio = (audio[:, 0]+audio[:, 1])/2
			
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

			
			return audio, label.astype('float32')


def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None, model='resnet18'):
	if split == 'train':
		shuffle = True
	else:
		shuffle = False
	data_loader = data.DataLoader(dataset=FMA(root, split=split, input_length=input_length, model=model),
								  batch_size=batch_size,
								  shuffle=shuffle,
								  drop_last=False,
								  num_workers=num_workers)
	return data_loader

