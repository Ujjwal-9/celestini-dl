from scipy.io import wavfile
import librosa

class AudioHandler():
    def __init__(self, file_path):
        self.file_path = file_path

    def loadData(self):
        ''' Loads the audio file '''
        samps,sr = librosa.load(self.file_path, mono=True, sr=None)
        self.sampling_rate = sr
        self.samples = samps

    def calc_features(self):
        ''' Calculates the auido features'''
        self.sample_count = len(self.samples)
        mfcc = librosa.feature.mfcc(self.samples, sr = self.sample_count)
