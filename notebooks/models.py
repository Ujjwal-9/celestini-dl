import keras
import kapre
from keras.models import Sequential
from keras.layers import Dense, AveragePooling2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Conv2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from optimizers import AdamW

import warnings
warnings.filterwarnings("ignore")


# 6 channels (!), maybe 1-sec audio signal, for an example.
input_shape = (1,16000)
sr = 16000

def depth_separable_cnn(input_shape=(1,16000), sr=16000, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam()):
	model = Sequential()
	# A mel-spectrogram layer
	model.add(Melspectrogram(n_dft=512, n_hop=512, input_shape=input_shape,
	                         padding='same', sr=sr, n_mels=128,
	                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
	                         return_decibel_melgram=True,trainable_fb=False,
	                         trainable_kernel=False,
	                         name='trainable_stft'))
	# Maybe some additive white noise.
	model.add(AdditiveNoise(power=0.1))
	# If you wanna normalise it per-frequency
	model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
	# After this, it's just a usual keras workflow. For example..
	# Add some layers, e.g., model.add(some convolution layers..)
	# Compile the model
	model.add(Conv2D(64, kernel_size=(20, 8), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
	model.add(Dropout(0.25))
	## Depth Seprable Pooling Layer - start
	model.add(SeparableConv2D(64, kernel_size=(5, 5), activation='relu',dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(1, 1), activation='relu',dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(SeparableConv2D(64, kernel_size=(5, 5), activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
	model.add(BatchNormalization())
	## Depth Seprable pooling Layer - end
	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(12, activation='softmax'))
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model

def esna(input_shape=(1,16000), sr=16000, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam()):
	model = Sequential()
	# A mel-spectrogram layer
	model.add(Melspectrogram(n_dft=512, n_hop=512, input_shape=input_shape,
	                         padding='same', sr=sr, n_mels=128,
	                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
	                         return_decibel_melgram=True,trainable_fb=False,
	                         trainable_kernel=False,
	                         name='trainable_stft'))
	# Maybe some additive white noise.
	model.add(AdditiveNoise(power=0.1))
	# If you wanna normalise it per-frequency
	model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
	# After this, it's just a usual keras workflow. For example..
	# Add some layers, e.g., model.add(some convolution layers..)
	# Compile the model
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(20, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(15, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(25, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(22, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(22, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(25, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(39, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu',dim_ordering="th"))

	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(12))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
	return model

def cnn(input_shape=(1,16000), sr=16000, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam()):
	model = Sequential()
	model.add(Melspectrogram(n_dft=512, n_hop=512, input_shape=input_shape,
                         padding='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
                         return_decibel_melgram=True,trainable_fb=False,
                         trainable_kernel=False,
                         name='trainable_stft'))
	# Maybe some additive white noise.
	model.add(AdditiveNoise(power=0.1))
	# If you wanna normalise it per-frequency
	model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
	# After this, it's just a usual keras workflow. For example..
	# Add some layers, e.g., model.add(some convolution layers..)
	# Compile the model

	model.add(Conv2D(64, kernel_size=(20, 8), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=(10, 4), activation='relu',dim_ordering="th"))
	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(12, activation='softmax'))
	model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
	return model

def esnd(input_shape=(1,16000), sr=16000, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam()):
	model = Sequential()
	# A mel-spectrogram layer
	model.add(Melspectrogram(n_dft=512, n_hop=512, input_shape=input_shape,
	                         padding='same', sr=sr, n_mels=128,
	                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
	                         return_decibel_melgram=True,trainable_fb=False,
	                         trainable_kernel=False,
	                         name='trainable_stft'))
	# Maybe some additive white noise.
	model.add(AdditiveNoise(power=0.1))
	# If you wanna normalise it per-frequency
	model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
	# After this, it's just a usual keras workflow. For example..
	# Add some layers, e.g., model.add(some convolution layers..)
	# Compile the model
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(33, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(35, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu',dim_ordering="th"))

	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Dense(12, activation='softmax'))
	model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
	return model

def esnb(input_shape=(1,16000), sr=16000, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam()):
	model = Sequential()
	# A mel-spectrogram layer
	model.add(Melspectrogram(n_dft=512, n_hop=512, input_shape=input_shape,
	                         padding='same', sr=sr, n_mels=128,
	                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
	                         return_decibel_melgram=True,trainable_fb=False,
	                         trainable_kernel=False,
	                         name='trainable_stft'))
	# Maybe some additive white noise.
	model.add(AdditiveNoise(power=0.1))
	# If you wanna normalise it per-frequency
	model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
	# After this, it's just a usual keras workflow. For example..
	# Add some layers, e.g., model.add(some convolution layers..)
	# Compile the model
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(9, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(11, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(10, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(11, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(30, kernel_size=(3, 3), activation='relu',dim_ordering="th"))
	model.add(Conv2D(45, kernel_size=(3, 3), activation='relu',dim_ordering="th"))

	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	# model.add(Dense(16))
	model.add(Dense(12, activation='softmax'))
	model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
	return model