#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:24:27 2019

@author: ninn
"""
def imgBatchDist(a,b):
#    assert type(a) == type(holder)
#    assert type(b) == type(holder)
#    assert a.get_shape() == b.get_shape()
    a_shape = a.shape
    b_shape = b.shape
    assert a_shape == b_shape
    batch_size = a_shape[0]
    d = 0.0
    for i in range(batch_size):
        d += imgDist(a[i], b[i])
    
    return d

def imgDist(a,b):
    a_shape = a.shape
    b_shape = b.shape
    assert a_shape == b_shape
    shape = a_shape
    d = 0.0
    for c in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                d + abs(a[i,j,c]-b[i,j,c])/255
                
    return d