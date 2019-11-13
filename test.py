#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:34:00 2019

@author: ninn
"""
import tensorflow as tf
import model
import data_provider
batch_size = 50
provider = data_provider.CISIAV1NormalizedImgProvider(batch_size)
provider.loadData()

save_rdir = './save/save'
saver = tf.train.Saver()

import utils
with tf.Session() as sess:
    utils.restoreModel(saver, save_rdir,sess)
    for xtest,basenametest,done in provider.nextTestOne():
        if done:
            break
        print(basenametest)
        print(xtest.shape)