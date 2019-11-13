#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:07:29 2019

@author: ninn
"""

import os
def mkrpath(rpath):
    print("[.] util.mkrpath")
    print("rpath:", rpath)
    rpaths = rpath.split("/")
    tmp = ""
    for rpath in rpaths:
        tmp = os.path.join(tmp,rpath)
        if not os.path.exists(tmp):
            os.mkdir(tmp)
            
    print("[o] util.mkrpath")
    return

def mkapath(apath):
    
    print("apath:", apath)
    if os.path.exists(apath):
        return
    print("[.] util.mkapath")
    apaths = apath.split("/")
    tmp = "/"
    for apath in apaths:
        if apath == '':
            continue
        tmp = os.path.join(tmp,apath)
        if not os.path.exists(tmp):
            os.mkdir(tmp)
            
    print("[o] util.mkrpath")
    return

def saveModel(saver, rdir, sess):
    import os
    rpath = os.path.dirname(rdir)
    mkrpath(rpath)
    saver.save(sess, rdir, write_meta_graph=False)
    return

def restoreModel(saver, rdir, sess):
    saver.restore(sess, rdir)
    return