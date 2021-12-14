# coding: utf-8
import os
import numpy as np
from torch.utils import data
import random
import soundfile as sf
import sys 
import tqdm
sys.path.append('..')
from torchvggish.vggish_input import waveform_to_examples
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch


def extract_spectrogram():
	files = open(f"../../../data/gtzan/valid_filtered.txt", 'r').readlines()
	for file in tqdm.tqdm(files):
		file = file.strip()
		audio, sr = sf.read(f'../../../data/gtzan/GTZAN/genres/{file}')
		audio = waveform_to_examples(audio, sr, return_tensor=False)
		print(audio.shape)
		np.save(f'../../../data/gtzan/vgg_spec/{file.split("/")[-1].replace("wav", "npy")}', audio)
#extract_spectrogram()

class GTZAN(data.Dataset):
	def __init__(self, root, split, input_length=None, model='resnet18'):
		split = split.lower()
		self.mappeing = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
		self.files = open(f"{root}/{split}_filtered.txt", 'r').readlines()
		self.class_num = 10
		self.split = split
		self.seg_length = input_length
		self.root = root
		self.model = model
		if self.model == 'hubert_ks':
			self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")


	def __len__(self):
		if self.split == 'train':
			return 100
		else:
			return len(self.files)

	def __getitem__(self, idx):
		if self.split == 'train':
			idx = random.randint(0, len(self.files)-1)
		file = self.files[idx].strip()
		frame = sf.info(os.path.join(self.root, file)).frames
		label = np.zeros(self.class_num)
		label[self.mappeing[file.split('/')[0]]] = 1
		if self.split == 'train':
			start = random.randint(0, frame-self.seg_length-16000)
			
			end = start + self.seg_length
			audio, sr = sf.read(os.path.join(self.root, file), start=start, stop=end)
			audio = audio.astype('float32')
			
			#vgg_start = int(round(start/16000/0.96))
			#audio = np.load(f'{self.root}/vgg_spec/{file.split("/")[1].replace("wav", "npy")}')[vgg_start: vgg_start+int(self.seg_length/16000)]

			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
			return audio, label.astype('float32')
		else:
			audio, sr = sf.read(os.path.join(self.root, file))
			audio = audio.astype('float32')
			
			n_chunk = len(audio) // self.seg_length 
			
			'''
			audio_chunks = np.load(f'{self.root}/vgg_spec/{file.split("/")[1].replace("wav", "npy")}')
			'''
			if self.model == 'hubert_ks':
				audio = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
				for d in audio.keys():
					audio[d] = audio[d][0]
					audio_chunks = torch.stack(torch.split(audio[d][:int(n_chunk*self.seg_length)], n_chunk))
					torch.cat((audio_chunks, audio[d][-int(self.seg_length):][:, None]), -1)
					audio[d] =  audio_chunks.T
				
			elif 'resnet' in self.model:
				audio_chunks = np.split(audio[:int(n_chunk*self.seg_length)], n_chunk)
				audio_chunks.append(audio[-int(self.seg_length):])
				audio = np.array(audio_chunks)

			return audio, label.astype('float32')


def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None, model='resnet18'):
	data_loader = data.DataLoader(dataset=GTZAN(root, split=split, input_length=input_length, model=model),
								  batch_size=batch_size,
								  shuffle=True,
								  drop_last=True,
								  num_workers=num_workers)
	return data_loader

