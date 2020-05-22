import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Flatten,MaxPool2D,MaxPool1D,Activation,LeakyReLU,LSTM,BatchNormalization,Dropout,Conv2D,Conv1D,Lambda
from keras.constraints import max_norm
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

def baseline_lstm():
	
	inputs = Input((128,2,))
	l = BatchNormalization()(inputs)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)   
	l = LSTM(128,return_sequences=False,activation='tanh',unroll=True)(l)
	l = Dropout(0.2)(l)	
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)
	
	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

def new_lstm():
	
	inputs = Input((128,4,))
	l = BatchNormalization()(inputs)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)   
	l = LSTM(128,return_sequences=False,activation='tanh',unroll=True)(l)
	l = Dropout(0.2)(l)
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

def baseline_conv():
	
	inputs = Input((128,2,))
	l = BatchNormalization()(inputs)
	l = Lambda(lambda t: K.expand_dims(t, -1))(l)
	l = Conv2D(filters=256,kernel_size=(3,1),activation='relu')(l)
	l = Conv2D(filters=80,kernel_size=(3,2),activation='relu')(l)
	l = Flatten()(l)    
	l = Dense(256,activation='relu')(l)
	l = Dropout(0.6)(l)     
        
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)
    
	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model
  

def base_scrnn():
	
	inputs = Input((128,2,))
	l = BatchNormalization()(inputs)
	# l = Lambda(lambda t: K.expand_dims(t, -2))(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l = MaxPool1D(3)(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	# l = Lambda(lambda t: K.squeeze(t, -2))(l)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)    
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)
	l = Dropout(0.5)(l)     
	l = Flatten()(l)    
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

def new_scrnn():
	
	inputs = Input((128,4,))
	l = BatchNormalization()(inputs)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l = MaxPool1D(3)(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)   
	l = LSTM(128,return_sequences=True,activation='tanh',unroll=True)(l)
	l = Dropout(0.5)(l)     
	l = Flatten()(l)    
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	model.summary()
	return model

