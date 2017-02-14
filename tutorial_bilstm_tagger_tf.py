from collections import Counter, defaultdict
from itertools import count
import random

import dynet as dy
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="WSJ_TRAIN"
dev_file="WSJ_DEV"

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
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
dev=list(read(dev_file))
words=[]
tags=[]
chars=set()
wc=Counter()
for sent in train:
    for w,p in sent:
        words.append(w)
        tags.append(p)
        chars.update(w)
        wc[w]+=1
words.append("_UNK_")
chars.add("<*>")

vw = Vocab.from_corpus([words]) 
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
nchars  = vc.size()

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

# Network building (word-level)
net = input_data(shape=[None, 128])
net = embedding(net, input_dim=nwords, output_dim=32)
net = bidirectional_rnn(net, BasicLSTMCell(32), BasicLSTMCell(32))
net = dropout(net, 0.5)
net = fully_connected(net, ntags, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
#model.fit(train, trainY, validation_set=(testX, testY), show_metric=True,
#          batch_size=32)



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

