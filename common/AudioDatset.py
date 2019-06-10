import os
import librosa
import numpy as np

class AudioFeatureDataset():
    ''' Creates Audio Dataset and extracts features from RAW wav files '''
    def __init__(self,file_path):
        self.file_path = file_path
        self.labels = os.listdir(self.file_path)
        self.target_labels = ['no', 'seven', 'right', 'up', 'down', 'eight', 'six', 'wow', 'bird', 'tree', 'happy', 'three', 'five', 'zero', 'go', 'left', 'nine', 'two', 'four', 'yes', 'bed', 'stop', 'cat', 'dog', 'marvin', 'off', 'one', 'on', 'sheila', 'house']
        self.data_dict = {}
        for tl in self.target_labels:
            files_dir = os.path.join(file_path,tl)
            all_audio_fp_s =[ os.path.join(files_dir,f) for f in os.listdir(files_dir)]
            self.data_dict[tl] = all_audio_fp_s
    
    def process(self,file,max_pad = 35):
        samps,sr = librosa.load(file, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(samps, sr = sr)
        pad_width = max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc
    def get_features(self):
        labels =  []
        features = []
        for t in self.target_labels:
            for fp in self.data_dict[t]:
                labels.append(t)
                features.append(self.process(fp))
        return labels, features