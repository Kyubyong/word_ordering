# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Sept. 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/word_ordering
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_data
from train import Graph
import distance
import glob
from tqdm import tqdm

np.random.seed(0)
def eval(): 
    # Load graph
    g = Graph(mode="test")
    print("Graph loaded")

    # Load batch
    _Y = load_data(mode="test")

    X = np.zeros((len(_Y), hp.maxlen))
    Y = np.zeros((len(_Y), hp.maxlen))
    for i, y in enumerate(_Y):
        y = np.fromstring(y, np.int32)
        Y[i][:len(y)] = y
        np.random.shuffle(y)
        X[i][:len(y)] = y

    word2idx, idx2word = g.word2idx, g.idx2word
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                num_words, total_edit_distance = 0, 0
                for i in range(0, len(Y), hp.batch_size):
                    ### Get mini-batches
                    x = X[i:i+hp.batch_size]
                    y = Y[i:i+hp.batch_size]

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    for xx, yy, pred in zip(x, y, preds):  # sentence-wise
                        inputs = " ".join(idx2word[idx] for idx in xx).replace("_", "").strip()
                        expected = " ".join(idx2word[idx] for idx in yy).replace("_", "").strip()
                        got = " ".join(idx2word[idx] for idx in pred[:len(inputs.split())])

                        edit_distance = distance.levenshtein(expected.split(), got.split())
                        total_edit_distance += edit_distance
                        num_words += len(expected.split())

                        fout.write(u"Inputs  : {}\n".format(inputs))
                        fout.write(u"Expected: {}\n".format(expected))
                        fout.write(u"Got     : {}\n".format(got))
                        fout.write(u"WER     : {}\n\n".format(edit_distance))
                fout.write(u"Total WER: {}/{}={}\n".format(total_edit_distance,
                                                           num_words,
                                                        round(float(total_edit_distance) / num_words, 2)))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    