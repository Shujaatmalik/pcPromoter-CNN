
import os
import sys

from focal_loss import BinaryFocalLoss
import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,AveragePooling1D
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers.wrappers import Bidirectional, TimeDistributed
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import Model
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Dense, Add
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import auc
from tensorflow.keras.layers import LSTM
from focal_loss import BinaryFocalLoss
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import StratifiedKFold

def get_model():

	input_shape = (81,4)
	inputs = Input(shape = input_shape)

	convLayer = Conv1D(filters = 32, kernel_size = 7,activation = 'relu',input_shape = input_shape, kernel_regularizer = regularizers.l2(1e-5), bias_regularizer = regularizers.l2(1e-4))(inputs)
	normalizationLayer = BatchNormalization()(convLayer);
	poolingLayer = AveragePooling1D(pool_size = 2, strides=2)(normalizationLayer)
	dropoutLayer0 = Dropout(0.35)(normalizationLayer)
    
	convLayer2 = Conv1D(filters = 32, kernel_size = 5,activation = 'relu',kernel_regularizer = regularizers.l2(1e-4), bias_regularizer = regularizers.l2(1e-5))(dropoutLayer0)
	poolingLayer2 = MaxPooling1D(pool_size = 2, strides=2)(convLayer2)
	dropoutLayer1 = Dropout(0.30)(poolingLayer2)
    
	flattenLayer = Flatten()(dropoutLayer1)

	denseLayer = Dense(16, activation = 'relu',kernel_regularizer = regularizers.l2(1e-3),bias_regularizer = regularizers.l2(1e-3))(flattenLayer)
	outLayer = Dense(1, activation='sigmoid')(denseLayer)

	model = Model(inputs = inputs, outputs = outLayer)
	model.compile(loss='binary_crossentropy', optimizer= SGD(momentum = 0.95, lr = 0.007), metrics=['binary_accuracy'])
	
	return model


modelProMN = get_model()
modelProMN.load_weights('best_weights_Pro_NonPro.h5')
modelSigma70=get_model()
modelSigma70.load_weights('best_weights_Sigma70.h5')
modelSigma24=get_model()
modelSigma24.load_weights('best_weights_Sigma24_v1.h5')
modelSigma28=get_model()
modelSigma28.load_weights('best_weights_Sigma28.h5')
modelSigma38=get_model()
modelSigma38.load_weights('best_weights_Sigma38.h5')
modelSigma32=get_model()
modelSigma32.load_weights('best_weights_Sigma32.h5')
import numpy as np


def encode_seq(s):
    Encode = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
    return np.array([Encode[x] for x in s])

X1 = {}
accumulator=0



sequences=[]
for record in SeqIO.parse("Independent test dataset/Sigma70Positive.txt", "fasta"):
    s=record.seq._data
    sequences.append(s)
  
Strng="TTCAGTGATAATTATCACATTTCAATTGCACATTAATGGATATTCTTTAATAATCTCGCGACGTTTCTTTATGATAAATAA"


def pcPromoter(Strng):
    X1 = {}
    my_hottie = encode_seq((Strng))
    out_final=my_hottie
     # out_final=out_final.astype(int)
    out_final = np.array(out_final)
    X1[accumulator]=out_final
      #out_final=list(out_final)
    X1[accumulator] = out_final
    
    X1 = list(X1.items()) 
    an_array = np.array(X1)
    an_array=an_array[:,1]
    
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X1=np.transpose(transpose_list)
    X1=np.transpose(X1)
    pr=modelProMN.predict(X1)
    pr=pr.round()
    if(pr==1):
        print('\n promoter \n')
        prS70=modelSigma70.predict(X1)
        prS70=prS70.round()
        if(prS70==1):
            print('\n sigma70 \n')
   
        else:
            pr24=modelSigma24.predict(X1)
            pr24=pr24.round()
            if(pr24==1):
                print('\n sigma24 \n')
        
            else:
                pr28=modelSigma28.predict(X1)
                pr28=pr28.round()
                if(pr28==1):
                    print('\n Sigma28 \n')
             
                else:
                    pr38=modelSigma38.predict(X1)
                    pr38=pr38.round()
                    if(pr38==1):
                        print('\n Sigma38 \n')
           
                    else:
                        pr32=modelSigma32.predict(X1)
                        pr32=pr32.round()
                        if(pr32==1):
                            print('\n Sigma32 \n')
    
                        else:
                            print('\n Sigma 54 \n')
 
                        
           
    else:
        print('non promoter')




   
for i in range(len(sequences)):
    pcPromoter(sequences[i])
 
  


