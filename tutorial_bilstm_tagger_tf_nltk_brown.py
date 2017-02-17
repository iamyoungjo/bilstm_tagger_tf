from collections import Counter, defaultdict
from itertools import count
import random

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

import sys
import numpy as np

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = sys.argv[1]
valid_rate = float(sys.argv[2])
#test_file="tagged_test.txt"


#Pre-processing

def read(fname):
    """
    Read a tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname) as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("/",1)) for x in line]
            yield sent


train=list(read(train_file))
words=[]
tags=[]
chars=set()
tagset = set()
for sent in train:
    for w,p in sent:
        words.append(w)#.lower())
        if(len(w)>25):
            print("over 25!", w)
        tags.append(p)
        tagset.add(p)
        chars.update(w)#.lower())


def w2v(word, d):
	wvec = []
	for c in word:
		wvec.append(d[c])
	return wvec

dictionary = {i:j for j,i in enumerate(list(chars))}
tagdict = {i:j for j,i in enumerate(list(tagset))}

trainW = []
for word in words:
	trainW.append(w2v(word, dictionary))
trainW = np.array(trainW)

trainT = []	
for tag in tags:
	trainT.append(tagdict[tag])
trainT = np.array(trainT)


charvocab = len(chars)
maxbound = 25
valid = int(len(words) * (1.0-valid_rate))

# Sequence padding
trainW = pad_sequences(trainW, maxlen=maxbound, value=0.)
testW = trainW[-valid:]
trainW = trainW[:-valid]
# Converting labels to binary vectors
trainT = to_categorical(trainT, nb_classes=len(tagset))
testT = trainT[-valid:]
trainT = trainT[:-valid]


# TFlearn starts
# Network building (char-level)
net = input_data(shape=[None, maxbound])
net = embedding(net, input_dim=charvocab, output_dim=256)
net = bidirectional_rnn(net, BasicLSTMCell(256), BasicLSTMCell(256))
net = dropout(net, 0.5)
net = fully_connected(net, len(tagset), activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# Training



model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainW, trainT, n_epoch=1000, validation_set=(testW, testT), show_metric=True, batch_size=512)

# ##example prediction after model set
# print("testW[0]: ", testW[0])
# print("Prediction of testW[0]: ", model.predict(testW[0]))

modelfile = ".."
#for i in range(10):
    #print( np.argmax(model.predict(testW[:10])[i]))

def i2t(index):
    return list(tagset)[index]
def is2w(indices):
    switch = 0
    word = []
    for i in range(maxbound):
        if(indices[i]!=0):
            switch = 1
        if(switch == 1):
            word.append(list(chars)[indices[i]])
    
    return "".join(word)    
    
#print([i2t(np.argmax(a)) for a in testT[:20]])
#print([i2t(np.argmax(a)) for a in model.predict(testW[:20])])
window = 80
print("Input:")
print(" ".join([is2w(testW[:window][i]) for i in range(len(testW[:window]))]))
print("\nPrediction: ")
print([is2w(testW[:window][i])+"/" +i2t(np.argmax(a)) for i, a in enumerate(model.predict(testW[:window]))])
print("\nAnswer: ")
print([is2w(testW[:window][i])+"/" +i2t(np.argmax(a)) for i, a in enumerate(testT[:window])])
model.save("trained_model_" + sys.argv[1] + "_" + sys.argv[1] + ".model")


