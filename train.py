#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:38:27 2019

@author: ninn
"""

import tensorflow as tf
import model
import data_provider
#import loss.triplet_loss as loss
batch_size = 50
provider = data_provider.CISIAV1NormalizedImgProvider(batch_size)
provider.loadData()

x_holder = tf.placeholder(shape=[None, provider.height, provider.width,provider.channels], dtype=tf.float32)
y_holder = tf.placeholder(shape=[None],dtype=tf.uint8)

layers = model.unet_triplet_loss(x_holder,y_holder)
feature = layers["feature"]
mask = layers["mask"]
triplet_loss = layers["all_triplet_loss"]
#embeddings = tf.layers.flatten(feature, name="embeddings")


#triplet_loss, fraction_postive_triplets = loss.batch_all_triplet_loss(y_holder,embeddings,0.5)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(triplet_loss)
init_op = tf.global_variables_initializer()

epochs = 2


save_rdir = './save/save'
saver = tf.train.Saver()

import utils


with tf.Session() as sess:
    sess.run(init_op)
    
    
    for i in range(epochs):
        x = None
        y = None
        done = False
        feed_dict = None
        while not done:
            
            x,y,done = provider.nextTrainBatch()
#            print(len(x))
#            print(x[0].shape)
#            print(len(y))
#            print(y)
#            print(done)
            feed_dict = {
                    x_holder:x,
                    y_holder:y
            }
            sess.run(train_op, feed_dict)
            
        
        loss = sess.run(triplet_loss, feed_dict)
        print("epoch {} loss: {}".format(i, loss))
#        print(loss)
        
    utils.saveModel(saver, sess=sess, rdir=save_rdir)
    
    
    xtestbatch,ytestbatch,basenametestbatch = provider.nextTestBatch()
    basenametest = basenametestbatch[0]
    feed_dict = {x_holder:xtestbatch}
    featuretestbatch = sess.run(feature, feed_dict)
    featuretest = featuretestbatch[0]
    print(featuretest)
    for x,basename,done in provider.nextTestOne():
        
        if done:
            break
        feed_dict = {x_holder:x}
        outs = sess.run(feature, feed_dict)
        for out in outs:
            print(out)
            exit(0)
#            from loss import loss
#            print("{} {} {}".format(basenametest, basename, loss.imgDist(featuretest,out[0])))
    
    
        
    