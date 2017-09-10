# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
Sept. 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/word_ordering
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import re
from collections import Counter
import random
import tensorflow as tf

random.seed(100)

def normalize(sent):
    '''
    >> sent = "“What gives you the right to judge?” she continued, then softened."
    >> normalize(sent)
    what gives you the right to judge she continued then softened
    '''
    sent = sent.lower()
    sent = re.sub("[^ a-z'\-]", "", sent)
    return sent


def load_vocab():
    words = []
    for line in codecs.open(hp.data, 'r', 'utf-8'):
        _, sent = line.strip().split("\t")
        sent = normalize(sent)
        words.extend(sent.split())
    vocab = []
    word2cnt = Counter(words)
    for word, cnt in word2cnt.most_common(len(word2cnt)):
        if cnt < hp.min_cnt: break
        vocab.append(word)
    vocab.sort()
    vocab = ["_"] + vocab

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    print("vocabulary size =", len(word2idx))
    return word2idx, idx2word

def load_data(mode="train"):
    word2idx, idx2word = load_vocab()

    Y = []
    for line in codecs.open(hp.data, 'r', 'utf-8'):
        _, sent = line.strip().split("\t")
        sent = normalize(sent)
        words = sent.split()
        if len(words) <= hp.maxlen:
            sent_ids = [word2idx.get(word, 0) for word in words]
            if 0 not in sent_ids: # We do not include a sentence if it has any unknown words.
                Y.append(np.array(sent_ids, np.int32).tostring())

    random.shuffle(Y)
    if mode=="train":
        Y = Y[:-100*hp.batch_size]
    else: # test
        Y = Y[-100*hp.batch_size:]

    print("# Y =", len(Y))
    return Y

def get_batch_data():
    with tf.device("/cpu:0"):
        # Load data
        Y = load_data(mode="train")

        # calc total batch count
        num_batch = len(Y) // hp.batch_size

        # Convert to tensor
        Y = tf.convert_to_tensor(Y)

        # Create Queues
        y, = tf.train.slice_input_producer([Y,])

        # Restore to int32
        y = tf.decode_raw(y, tf.int32)

        # Random generate inputs
        x = tf.random_shuffle(y)

        # create batch queues
        x, y = tf.train.batch([x, y],
                              num_threads=8,
                              batch_size=hp.batch_size,
                              capacity=hp.batch_size * 64,
                              allow_smaller_final_batch=False,
                            dynamic_pad=True)

    return x, y, num_batch  # (N, T), ()
#
