from datapreprocessing import *
from model import *
import pandas as pd
import numpy as np

def main(file_train = 'train.csv', file_test = 'test.csv', file_pred = 'pred.csv'):

	train_data = get_data(file_train)
	human_vocab = train_data['human_vocab']
	human_dict = train_data['human_dict']
	one_hot_train = train_data['one_hot_statements']
	one_hot_eqn = train_data['one_hot_eqn']
	numeric_eqn, num_vocab, num_dict = train_data['numeric_eqn'], train_data['num_vocab'], train_data['num_dict']
	test_data = get_data(file_test, train=False, human_vocab=human_vocab, human_dict = human_dict) 
	Tx, Ty = one_hot_train.shape[1], one_hot_eqn.shape[1]

	model = Model(32, 64, num_vocab, human_vocab, Tx, Ty)

	s0 = np.zeros((one_hot_train.shape[0], n_s))
	c0 = np.zeros((one_hot_train.shape[0], n_s))
	Y = list(one_hot_eqn.swapaxes(0,1))
	model.fit([one_hot_train, s0, c0], Y, epochs=120, shuffle=True)

	inv_y_dict = {i : num_vocab[i] for i in range(len(human_vocab))}


	one_hot_test = test_data['one_hot_statements']
	stest = np.zeros((one_hot_test.shape[0], n_s))
	ctest = np.zeros((one_hot_test.shape[0], n_s))
	predictions = model.predict([one_hot_test, stest, ctest])
	predictions = np.argmax(predictions, axis=-1)
	predictions = predictions.T
	soln = np.zeros((one_hot_test.shape[0],Ty),dtype=str)
	for i in range(one_hot_test.shape[0]):
		for j in range(Ty):
			soln[i,j] = inv_y_dict[predictions[i,j]]


	fir = np.zeros(80,)
	sec = np.zeros(80,)
	operation = np.aget_dataay(soln[:,2],dtype=str)
	state=0
	for i in range(80):
		num1=0
		num2=0
		sign=None
		for j in range(5):
			if state==0 and soln[i,j].isdigit():
				num1 = num1*10 + int(soln[i,j])
    		elif state==1 and soln[i,j].isdigit():
				num2 = num2*10 + int(soln[i,j])
    		elif soln[i,j] in ["+", "-", "*", "^"]:
				state=1
				sign=soln[i,j]
		fir[i]=num1
		sec[i]=num2
		operation[i]=sign
		state=0

	epn = []
	for i in range(80):
		epn.append("x" + operation[i] + "y")

	test_df = pd.read_csv(file_test)
	test_df["Significant number 1 detected"] = fir
	test_df["Significant number 2 detected"] = sec
	test_df["equation generated"] = epn
	test_df.to_csv(file_pred)



if __name__ == '__main__' :
	main()







	


