from torch.utils import data
import json
import numpy as np
import torch
import soundfile as sf
import os
import random

class singer32(data.Dataset):
	def __init__(self, root, split, input_length=None, model='resnet18'):
		split = split.lower()

		if split == 'train':
			query = json.load(open ('../../data/singer32/singer32_train.json', 'r'))
		elif split == 'test':
			query = json.load(open ('../../data/singer32/singer32_test.json', 'r'))
		else:
			query = json.load(open ('../../data/singer32/singer32_validate.json', 'r'))
		
		self.class_num = 32
		self.files, self.singers = [], []
		audio_files = os.listdir('../../data/singer32/wav_16000/')
		for i, k in enumerate(query):
			files = [file for file in query[k] if file+'.wav' in audio_files]
			self.singers += [i for j in range(len(files))]
			self.files += files 

		self.split = split
		self.seg_length = input_length
		self.root = root
		self.model = model
		if self.model == 'hubert_ks':
			self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")


	def __len__(self):
		if self.split == 'train':
			return 10000 #len(self.files)
		else:
			return len(self.files)

	def __getitem__(self, idx):
		
		if self.split == 'train':
			idx = random.randint(0, len(self.files)-1)
			file = self.files[idx]
			label = np.zeros(self.class_num)
			label[self.singers[idx]] = 1
			frame = sf.info(os.path.join(self.root, 'wav_16000', file+'.wav')).frames
			start = random.randint(0, frame-self.seg_length-1)
			end = start + self.seg_length
			audio, sr = sf.read(os.path.join(self.root, 'wav_16000', file+'.wav'), start=start, stop=end)
			audio = audio.astype('float32')
			if len(audio.shape) == 2:
				audio = (audio[:, 0]+audio[:, 1])/2
			
			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
			return audio, label.astype('float32')
			
		else:
			file = self.files[idx]
			label = np.zeros(self.class_num)
			label[self.singers[idx]] = 1

			audio, sr = sf.read(os.path.join(self.root, 'wav_16000', file+'.wav'))
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

	data_loader = data.DataLoader(dataset=singer32(root, split=split, input_length=input_length, model=model),
								  batch_size=batch_size,
								  shuffle=shuffle,
								  drop_last=False,
								  num_workers=num_workers)
	return data_loader

