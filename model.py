#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:28:48 2019

@author: ninn
"""
import tensorflow as tf

def unet_triplet_loss(x, y):
    from loss import triplet_loss
    layers = {}
    layers.update(unet(x))
    embeddings = tf.layers.flatten(layers["feature"], name="embeddings")
    print(embeddings.get_shape())
    layers["embeddings"] = embeddings
    layers.update(triplet_loss.batch_all_triplet_loss(y,embeddings,  0.5))
    return layers

def unet(x):
    
    layers = {}
#    x_shape = []
#    for dim in x.shape:
#        x_shape.append(dim)
#    x_shape[0] = None
#    x = tf.placeholder(shape = x_shape, dtype=tf.float32)
    
    stack = []
    
    with tf.variable_scope("conv11"):
    
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 24,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
        
        print(x.get_shape())
        stack.append(x)
    
    
    with tf.variable_scope("down1"):
    
        x = tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=2,
                    strides=(2,2),
                    padding='same'
                )
        
        print(x.get_shape())
    
    with tf.variable_scope("conv12"):
        
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 32,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
        
        print(x.get_shape())
        stack.append(x)
    
    with tf.variable_scope("down2"):
        
        x = tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=2,
                    strides=(2,2),
                    padding='same'
                )
        
        print(x.get_shape())
    
    with tf.variable_scope("conv13"):
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 48,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
        
        print(x.get_shape())
        stack.append(x)
    
    with tf.variable_scope("down3"):
        x = tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=2,
                    strides=(2,2),
                    padding='same'
                )
        
        print(x.get_shape())
    
    with tf.variable_scope("conv14"):
    
        for i in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 64,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
            
            print(x.get_shape())
                
    
    with tf.variable_scope("up1"):
        
        x = tf.layers.conv2d_transpose(
                inputs = x,
                filters = 48,
                kernel_size = 3,
                strides=(2, 2),
                padding='same',
            )
#    print(x.get_shape())
#    return
        
    with tf.variable_scope("concat1"):
        x = tf.concat(values = (stack.pop(), x), axis=3)
        print(x.get_shape())
     
    with tf.variable_scope("conv21"):
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 48,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
            
    with tf.variable_scope("up2"):
        x = tf.layers.conv2d_transpose(
                inputs = x,
                filters = 32,
                kernel_size = 3,
                strides=(2, 2),
                padding='same',
            )
        
    with tf.variable_scope("concat2"):
        x = tf.concat(values = (stack.pop(), x), axis=3)
        print(x.get_shape())
    
    with tf.variable_scope("conv22"):
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 32,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
    with tf.variable_scope("up3"):
        x = tf.layers.conv2d_transpose(
                inputs = x,
                filters = 24,
                kernel_size = 3,
                strides=(2, 2),
                padding='same',
            )
    with tf.variable_scope("concat3"):
        x = tf.concat(values = (stack.pop(), x), axis=3)
        print(x.get_shape())
        
    with tf.variable_scope("conv23"):
        for j in range(2):
            x = tf.layers.conv2d(
                    inputs = x,
                    filters = 24,
                    kernel_size = 3,
                    strides=(1, 1),
                    padding='same',
                )
        print(x.get_shape())
    
    
    feature = tf.layers.conv2d(
                inputs = x,
                filters = 1,
                kernel_size = 1,
                strides=(1, 1),
                padding='same',
                name = "feature"
            )
    
    
    mask = tf.layers.conv2d(
                inputs = x,
                filters = 1,
                kernel_size = 1,
                strides=(1, 1),
                padding='same',
                name = "mask"
            )
    
    layers["feature"] = feature
    layers["mask"] = mask
    
    return layers