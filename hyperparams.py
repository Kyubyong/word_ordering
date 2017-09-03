# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Sept. 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/word_ordering
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    data = 'eng_news_2015_1M/eng_news_2015_1M-sentences.txt'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    smoothing_rate = 0.
    
    
    
    
