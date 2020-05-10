import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Flatten,MaxPool2D,MaxPool1D,Activation,LeakyReLU,LSTM,BatchNormalization,Dropout,Conv2D,Conv1D,Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

def baseline_lstm():

	# Dropout and regularization not added. Make modifications to add those layers
	# change the hyperparameter values as mentioned in the cited paper

	inputs = Input((128,2,))
	l = BatchNormalization()(inputs)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)
	l = LSTM(128,return_sequences=False,activation='tanh',unroll=True)(l)
	# l = Flatten()(l)
	outputs = Dense(11,activation='softmax')(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

def base_scrnn():

	# Dropout and regularization not added. Make modifications to add those layers
	# change the hyperparameter values as mentioned in the cited paper

	inputs = Input((128,2,))
	l = BatchNormalization()(inputs)
	# l = Lambda(lambda t: K.expand_dims(t, -2))(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l = MaxPool1D(3)(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	# l = Lambda(lambda t: K.squeeze(t, -2))(l)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)
	l = Flatten()(l)
	outputs = Dense(11,activation='softmax')(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

def baseline_conv():

	# incomplete model. Code up the architecture as mentioned in the cited paper.

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

