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
train_file=sys.argv[1]
test_file=sys.argv[2]



class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(int)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(int)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname) as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("/",1)) for x in line]
            yield sent

train=list(read(train_file))
test=list(read(test_file))
words=[]
tags=[]
chars=set()
wc=Counter()
for sent in train:
    for w,p in sent:
        words.append(w)#float(w.split("word")[1]))
        tags.append(float(p.split("tag")[1]))
        chars.update(w)
        wc[w]+=1
# words.append("_UNK_")
# chars.add("<*>")


words_t=[]
tags_t=[]
chars_t=set()
wc_t=Counter()
for sent in test:
    for w,p in sent:
        words_t.append(w)
        tags_t.append(float(p.split("tag")[1]))
        chars_t.update(w)
        wc_t[w]+=1
# words_t.append("_UNK_")
# chars_t.add("<*>")

vw = Vocab.from_corpus([words]) 
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([chars])
# UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
nchars  = vc.size()

def w2v(word, d):
	wvec = []
	for c in word:
		wvec.append(d[c])
	return wvec

dictionary = {'0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, 'w' : 10, 'o' : 11, 'r' : 12, 'd' : 13}

trainW = []
for word in words:
	#wvec = np.zeros(14)
	#wvec[int(word)-1] = 1.0
	trainW.append(w2v(word, dictionary))
trainW = np.array(trainW)# .reshape(len(words), 50000)

trainT = []	
for tag in tags:
# 	tvec = np.zeros(10)
# 	tvec[int(tag)-1] = 1.0
	trainT.append(int(tag)-1)
trainT = np.array(trainT)#.reshape(len(tags), 1)

testW = []
for word in words_t:
# 	wvec = np.zeros(14)
# 	wvec[int(word)-1] = 1.0
	testW.append(w2v(word, dictionary))
testW = np.array(testW)# .reshape(len(words_t), 50000)

testT = []	
for tag in tags_t:
# 	tvec = np.zeros(10)
# 	tvec[int(tag)-1] = 1.0
	testT.append(int(tag)-1)
testT = np.array(testT)#.reshape(len(tags_t), 1)


#Pre-processing
# Sequence padding
trainW = pad_sequences(trainW, maxlen=20, value=0.)
testW = pad_sequences(testW, maxlen=20, value=0.)
# Converting labels to binary vectors
trainT = to_categorical(trainT, nb_classes=10)
testT = to_categorical(testT, nb_classes=10)


# DyNet Starts
# TFlearn starts

"""
model = dy.Model()
trainer = dy.AdamTrainer(model)

WORDS_LOOKUP = model.add_lookup_parameters((nwords, 128))
CHARS_LOOKUP = model.add_lookup_parameters((nchars, 20))
p_t1  = model.add_lookup_parameters((ntags, 30))

# MLP on top of biLSTM outputs 100 -> 32 -> ntags
pH = model.add_parameters((32, 50*2))
pO = model.add_parameters((ntags, 32))

# word-level LSTMs
fwdRNN = dy.LSTMBuilder(1, 128, 50, model) # layers, in-dim, out-dim, model
bwdRNN = dy.LSTMBuilder(1, 128, 50, model)

# char-level LSTMs
cFwdRNN = dy.LSTMBuilder(1, 20, 64, model)
cBwdRNN = dy.LSTMBuilder(1, 20, 64, model)
"""


# Data preprocessing
print("nwords: ", nwords)
print("ntags: ", ntags)
print("train[:5] = ", train[:5])

# Network building (word-level)
net = input_data(shape=[None, 20])
net = embedding(net, input_dim=14, output_dim=32)
net = bidirectional_rnn(net, BasicLSTMCell(32), BasicLSTMCell(32))
net = dropout(net, 0.5)
net = fully_connected(net, ntags, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# Training



model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainW, trainT, validation_set=(testW, testT), show_metric=True,
          batch_size=64)

##example prediction after model set
print("testW[0]: ", testW[0])
print("Prediction of testW[0]: ", model.predict(testW[0]))


"""
def word_rep(w, cf_init, cb_init):
    if wc[w] > 5:
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = vc.w2i["<*>"]
        char_ids = [pad_char] + [vc.w2i[c] for c in w] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])


def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt.w2i[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def tag_sent(words):
    vecs = build_tagging_graph(words)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(vt.i2w[tag])
    return zip(words, tags)

num_tagged = cum_loss = 0
for ITER in xrange(50):
    random.shuffle(train)
    for i,s in enumerate(train,1):
        if i > 0 and i % 500 == 0:   # print status
            trainer.status()
            print (cum_loss / num_tagged)
            cum_loss = num_tagged = 0
            num_tagged = 0
        if i % 10000 == 0 or i == len(train)-1: # eval on dev
            good_sent = bad_sent = good = bad = 0.0
            for sent in dev:
                words = [w for w,t in sent]
                golds = [t for w,t in sent]
                tags = [t for w,t in tag_sent(words)]
                if tags == golds: good_sent += 1
                else: bad_sent += 1
                for go,gu in zip(golds,tags):
                    if go == gu: good += 1
                    else: bad += 1
            print (good/(good+bad), good_sent/(good_sent+bad_sent))
        # train on sent
        words = [w for w,t in s]
        golds = [t for w,t in s]

        loss_exp =  sent_loss(words, golds)
        cum_loss += loss_exp.scalar_value()
        num_tagged += len(golds)
        loss_exp.backward()
        trainer.update()
    print ("epoch %r finished" % ITER)
    trainer.update_epoch(1.0)

"""

