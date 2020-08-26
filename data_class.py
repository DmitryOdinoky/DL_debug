import pandas as pd
import sklearn, sklearn.utils
import torch

from python_speech_features import mfcc
    
import numpy as np
import mmap

from tqdm import tqdm


import librosa
import json

import functools
import operator
            
            
class fsd_dataset(object):
    
    def __init__(self, csv_file, path, train = True):

        self.if_train = True
      
        self.dataset_frame = pd.read_csv(csv_file).dropna(axis=0, how='all')
        
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        #self.dataset_frame = self.dataset_frame.drop(['index'], axis=1)
      
        #self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame.label.unique()

        self.all_samples = []
        
        self.metadata_array = []
        self.spectrogram_array = []
      
        
        self.path = path

        #self.time_window = 140
        
        self.hop_length = 512
        self.n_fft = 1024

        self.samplerate = 44100

        self.window_length = 4410 # depending on sample length
        self.overlap_length = 1000

        self.n_mels = 26
        self.n_mfcc = 13

        self.samples = []

        self.y_by_label = {}
        self.y_counts = {}
        
        for file in tqdm(self.dataset_frame['fname']):

            raw, sr = librosa.load(self.path + file, sr=self.samplerate, mono=True)
            #raw = librosa.resample(raw, sr, self.samplerate)

            label = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label'].values[0]
            
            
            
            split_points = librosa.effects.split(raw, top_db=80, frame_length=self.n_fft, hop_length=self.hop_length)
            
            S_cleaned = []
                        
            for piece in split_points:
             
                S_cleaned.append(raw[piece[0]:piece[1]])
         
            S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
           
            for i in range(0,1):
                S = np.concatenate((S,S),axis=0)
                
                



            for idx in range(0, S.shape[0]-self.window_length, self.overlap_length):
                sample = S[idx : idx+self.window_length]
                
                x = librosa.feature.mfcc(
                    sample,
                    sr=self.samplerate,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )            
                
                # x_max = x.max()
                # x_min = x.min()
                # x -= x_min
                # x /= (x_max - x_min) # 0..1
                # x -= 0.5
                # x *= 2.0 # -1 .. 1          
                
                x = np.log(abs(x) + 1e-20)  
                
                      

                if label not in self.y_by_label:
                    self.y_by_label[label] = len(self.y_by_label)

                y = self.y_by_label[label]
                
                #self.samples.append((x, y))

                if y not in self.y_counts:
                    self.y_counts[y] = 0
                
                if self.y_counts[y] < 100:
                    self.samples.append((x,y))
                    self.y_counts[y] += 1
                
                
            # S = librosa.stft(
            #     raw,
            #     sr=self.samplerate,
            #     n_fft=self.n_fft,
            #     hop_length=self.hop_length
            # )


            
           
            


            # S_max = S.max()
            # S_min = S.min()
            # S -= S_min
            # S /= (S_max - S_min) # 0..1
            # S -= 0.5
            # S *= 2.0 # -1 .. 1



            #print(f'file: {file} label: {label} counts: {S.shape[1]} / {self.window_length}')
            # if len(self.samples) > 64:
            #     break

        print(f'len(self.samples): {len(self.samples)}')
        print('self.y_counts:')
        for key, y in self.y_by_label.items():
            print(f'{key}: {self.y_counts[y]}')

        self.y_weights = torch.zeros((len(self.y_counts.keys()),), dtype=torch.float)
        for key in self.y_counts.keys():
            self.y_weights[key] = 1.0 - self.y_counts[key] / len(self.samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return np.expand_dims(x, axis=0).astype(np.float32), y








