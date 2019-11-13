#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:33:40 2019

@author: ninn
"""

import tensorflow as tf
import model
import data_provider
import loss.triplet_loss as loss
batch_size = 50
provider = data_provider.CISIAV1NormalizedImgProvider(batch_size)
provider.loadData()

x_holder = tf.placeholder(shape=[None, provider.height, provider.width,provider.channels], dtype=tf.float32)
y_holder = tf.placeholder(shape=[None, 1],dtype=tf.float32)

tboard_dir = "log"

with tf.Session() as sess:
    layers = model.unet_triplet_loss(x_holder, y_holder)
    writer = tf.summary.FileWriter(tboard_dir, sess.graph)