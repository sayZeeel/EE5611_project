import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
from keras.models import load_model
import keras
import keras.backend as K 
from sklearn.metrics import classification_report, confusion_matrix

def get_snr_range():
	return np.array([-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18])

def plot_snr_evaluation(fig_file,SNR_range,val_accuracy):

	plt.plot(SNR_range,val_accuracy,label = 'validation_accuracy')
	plt.title('Accuracy vs SNRs')
	plt.xlabel('Accuracy')
	plt.xlabel('SNRs')
	plt.legend()
	plt.grid(True)
	plt.savefig(fig_file)

def evaluate_model(models_dir,modelfile,xtest,ytest,test_snr_sequence,plot=True):
	m = load_model(models_dir+modelfile)
	ypred = m.predict(xtest)
	SNR = get_snr_range()

	val_acc = []


	for snr in SNR:
		p = np.where(test_snr_sequence == snr)[0]
		y_gt = ytest[p]
		y_pd = ypred[p]

		gt_class =np.argmax(np.array(y_gt),axis = 1)
		pd_class =np.argmax(np.array(y_pd),axis = 1)

		acc = len(np.where(gt_class==pd_class)[0])*100/len(gt_class)
		val_acc.append(acc)

		print("===============================================================\n")
		print("SNR: {}dB, Validation Accuracy: {}\n".format(snr,acc))
		print("\n             Classification Report")
		print(classification_report(gt_class,pd_class))
		print("\n             Confusion Matrix")
		c = np.array(confusion_matrix(gt_class,pd_class))
		# c = np.round(c*100/len(gt_class),2)
		print(c)

	if plot == True:
		plot_snr_evaluation(models_dir + 'fig_val_acc.png',SNR,val_acc)

	return val_acc