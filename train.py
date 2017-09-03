# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/word_ordering
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_vocab
from modules import *
import os, codecs
from tqdm import tqdm
import matplotlib.pyplot as plt

class Graph():
    def __init__(self, mode="train"):
        self.graph = tf.Graph()
        is_training = mode=="train"
        with self.graph.as_default():
            self.x, self.y, self.num_batch = get_batch_data(mode=mode) # (N, T)

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1]), self.y[:, :-1]), -1)

            # Load vocabulary
            self.word2idx, self.idx2word = load_vocab()

            # Embedding
            with tf.variable_scope("embedding"):
                ## encoder embedding
                self.enc = embedding(self.x, 
                                      vocab_size=len(self.word2idx),
                                      num_units=hp.hidden_units, 
                                      scale=True)

                ## decoder embedding
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(self.word2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     reuse=True)

            # Encoder
            with tf.variable_scope("encoder"):
                ## Positional Encoding
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe") 
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc, _ = multihead_attention(queries=self.enc,
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            
            # Decoder
            with tf.variable_scope("decoder"):
                ## Positional Encoding
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec, _ = multihead_attention(queries=self.dec,
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec, self.alignments = multihead_attention(queries=self.dec,
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(self.word2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
            if is_training:  
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(self.word2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

def plot_alignment(alignment,gs):
    """
    Plots the alignment
    alignment: (numpy) matrix of shape (encoder_steps,decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im=ax.imshow(alignment, cmap="Greys",interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.savefig(hp.logdir+'/alignment_%d'%gs,format='png')

if __name__ == '__main__':
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    word2idx, idx2word = g.word2idx, g.idx2word

    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

                # monitoring
                if step % 100 == 0:
                    _x, _y, _preds, _alignments, _gs = sess.run([g.x, g.y, g.preds, g.alignments, g.global_step])
                    print("\ninput=", " ".join(idx2word[idx] for idx in _x[0]))
                    print("expected=", " ".join(idx2word[idx] for idx in _y[0]))
                    print("got=", " ".join(idx2word[idx] for idx in _preds[0]))
                    plot_alignment(_alignments[0], _gs)

                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
    print("Done")    
    

