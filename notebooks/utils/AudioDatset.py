import os
import librosa
import numpy as np
from tqdm import tqdm 

## data augmentation
def roll(data):
    data_roll = np.roll(data, 5000)
    return data_roll
def stretch(data, rate=2):
    input_length = 16000*3
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def wnoise(data):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn
    return data_wn


def get_all_files(datapath, dataset_type="keyword"):
    data_dir = datapath
    data = []
    all_files = os.listdir(data_dir)
    all_files.remove('.DS_Store')
    labels = set()
    for file in all_files:
        filelabels = file.split("-")[:3]
        data_dict = {
            "filepath": data_dir + file,
            "stress": filelabels[2],
            "environment": filelabels[1],
            "keyword":filelabels[0]
        }
        labels.add(data_dict[dataset_type])
        data.append(data_dict)

    return data,labels


class AudioFeatureDataset():

    ''' To create audio dataset
        @param dataset_type = ( keyword | stress | environment   )
    '''

    def __init__(self,datapath, samplingrate=16000, dt="keyword"):
        print(dt)
        datafiles, labels = get_all_files(datapath,dataset_type=dt)
        self.datafiles = datafiles
        self.samplingrate = samplingrate
        self.target_labels = list(labels)
        self.dataset_type = dt

    def process(self, file, max_len=16000):
        ''' extracts raw audio  and returns samps '''
        try:
            samps, sr = librosa.load(file, mono=True, sr=None)
            pad_len = max_len - samps.shape[0]
            if pad_len >= 0:
                samps = np.pad(samps, (0, pad_len), 'constant')
            return np.array(samps[:max_len])
        except:
            print(file)

    def get_dataset(self):
        ''' returns dataset with augmented data '''
        labels = []
        features = []
        for file_data in tqdm(self.datafiles):
            labels.append(file_data[self.dataset_type])
            samps = self.process(file_data["filepath"], self.samplingrate * 3)
            features.append(samps)
            # with roll
            labels.append(file_data[self.dataset_type])
            features.append(roll(samps)) 
            # with strech
            labels.append(file_data[self.dataset_type])
            features.append(stretch(samps)) 
            # white noise
            labels.append(file_data[self.dataset_type])
            features.append(wnoise(samps)) 
        labels = np.array(labels)
        features = np.array(features)
        return features, labels

