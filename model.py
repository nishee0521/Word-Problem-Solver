import numpy as np
from keras.layers import *
from keras.models import Model


Tx = 19
Ty = 5
def softmaxaxis1(x):
	return softmax(x, axis=1)


def one_step(a, s_prev):
	s_prev = repeator(s_prev)
	concat = concatenator([a,s_prev])
	e = densor1(concat)
	energies = densor2(e)
	alphas = activator(energies)
	context = dotor([alphas,a])
	return context


def Model(n_a,n_s, y_vocab, x_vocab, Tx,Ty):
	activator = Activation(softmaxaxis1)
	repeator = RepeatVector(Tx)
	concatenator = Concatenate(axis=-1)
	densor1 = Dense(10, activation = "tanh")
	densor2 = Dense(1, activation = "relu")
	dotor = Dot(axes = 1)

	post_activation_LSTM_cell = LSTM(n_s, return_state = True)
	output_layer = Dense(len(y_vocab), activation=softmaxaxis1)
	X = Input((Tx, len(x_vocab)))
	s0 = Input((n_s,))
	c0 = Input((n_s,))
	s=s0
	c=c0
	outputs = []
	a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
	for t in range(Ty):
		context = one_step(a,s)
		s,_,c = post_activation_LSTM_cell(context, initial_state = [s,c])
		out = output_layer(s)
		outputs.append(out)
	model = Model(inputs = [X,s0,c0], outputs = outputs)
	model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

	return model



	
	
