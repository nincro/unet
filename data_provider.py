#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:40 2019

@author: ninn
"""

class DataProvider:
    
    data_apath = ""
    
    xall = []
    yall = []
    len_all = 0
    
    xtrain = []
    ytrain = []
    len_train = 0
    
    xval = []
    yval = []
    len_val = 0
    
    xtest = []
    ytest = []
    len_test = 0
    
    idx = 0
    
    train_weight = 8
    val_weight = 0
    test_weight = 2
    batch_size = 200
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        return
    def nextTrainBatch(self):
        done = False
        nidx = self.idx+self.batch_size
        if nidx > self.len_train:
            nidx = self.len_train+1
            
        xbatch = self.xtrain[self.idx: nidx]
        ybatch = self.ytrain[self.idx: nidx]
        
        if nidx > self.len_train:
            self.idx = 0
            done = True
        else:
            self.idx = nidx
            
        return xbatch, ybatch, done
        
    def loadData(self):
        
        return
    
    pass

class ImgDataProvider(DataProvider):
    
    suffix = ".jpg"
    imglist_adir = ""
    
    train_weight = 8
    val_weight = 0
    test_weight = 2
    
    width = 0
    height = 0
    channels = 1
    
    data_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-Iris-Thousand/"
    
    div = 3
    
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        return
    
    def loadData(self):
        import os,glob,cv2
        ymap = {}
        num = 1
        if self.imglist_adir == "":
            
            absrdir = "*"+self.suffix
            absadir = os.path.join(self.data_apath, absrdir)
            adirs = glob.glob(absadir)
            for adir in adirs:
                basename = os.path.basename(adir).split(".")[0]
                y = basename[:self.div]
                if ymap.get(y) is None:
                    ymap[y] = num
                    num+=1
                y = ymap[y]
                x = cv2.imread(adir)
                
                
                self.xall.append(x)
                self.yall.append(y)
                self.len_all = len(self.xall)
        self.len_train = int(self.len_all/10*self.train_weight)
        print("self.len_train:",self.len_train)
        self.xtrain = self.xall[:self.len_train]
        self.ytrain = self.yall[:self.len_train]
        
        self.len_val = int(self.len_all/10*self.val_weight)
        print("self.len_val:",self.len_val)
        self.xval = self.xall[self.len_train:self.len_val]
        self.yval = self.yall[self.len_train:self.len_val]
        
        self.len_test = self.len_all-self.len_train-self.len_val
        print("self.len_test:",self.len_test)
        self.xtest = self.xall[self.len_train+self.len_val:]
        self.ytest = self.yall[self.len_train+self.len_val:]
        
        return
    pass

class GroupedImgDataProvider(ImgDataProvider):
    
    data_apath = ""
    suffix = ""
    div = 3
    
    xall = {}
    yall = {}
    len_all = 0
    classes_all = 0
    basenameall = {}
    
    xtrain = {}
    ytrain = {}
    xtest = {}
    ytest = {}
    basenametest = {}
    
    idxs = []
    origin_idxs = []
    idx = 0
    
    
    def loadData(self):
        import os,glob,cv2
        ymap = {}
        num = 0
        if self.imglist_adir == "":
            
            absrdir = "*"+self.suffix
            absadir = os.path.join(self.data_apath, absrdir)
            adirs = glob.glob(absadir)
            for adir in adirs:
                basename = os.path.basename(adir).split(".")[0]
                
                y = basename[:self.div]
                
                if ymap.get(y) is None:
                    ymap[y] = num
                    num+=1
                    self.classes_all+=1
                y = ymap[y]
                x = cv2.imread(adir)
                
                if self.xall.get(y) is None:
                    self.xall[y] = [x]
                else:
                    self.xall[y].append(x)
                    
                if self.yall.get(y) is None:
                    self.yall[y] = [y]
                else:
                    self.yall[y].append(y)
                    
                if self.basenameall.get(y) is None:
                    self.basenameall[y] = [basename]
                else:
                    self.basenameall[y].append(basename)
                    
                self.len_all+=1
            
            import numpy as np
            self.idxs = np.array(range(self.classes_all))
            self.origin_idxs = self.idxs
            
            self.xtest[0] = self.xall[0]
            self.ytest[0] = self.yall[0]
            self.basenametest[0] = self.basenameall[0]
        return
    
    
    def nextTrainBatch(self):
        done = False
        nidx = self.idx+2
        xbatch = []
        ybatch = []
        if nidx>self.classes_all:
            nidx=self.classes_all
        for i in range(self.idx,nidx):
            ybatch.extend(self.yall[self.idxs[i]])
            xbatch.extend(self.xall[self.idxs[i]])
        self.idx = nidx
        if self.idx >= self.classes_all:
            done = True
            self.idx = 0
            import numpy as np
            np.random.shuffle(self.idxs)
            
        return xbatch,ybatch,done
    
    def nextTestBatch(self):
        xbatch = self.xtest[0]
        ybatch = self.ytest[0]
        basenamebatch = self.basenametest[0]
        return xbatch,ybatch,basenamebatch
    
    def __init__(self):
        return
    
    def nextTestOne(self):
        done = False
        for idx in self.origin_idxs:
            for c in range(len(self.xall[idx])):
                yield ([self.xall[idx][c]],[self.basenameall[idx][c]],done)
        done = True
        return None,None,done

class ThousandImgProvider(ImgDataProvider):
    
   
    
    width = 640
    height = 480
    channels = 1
    div = 5
    
    data_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-Iris-Thousand/"
    
class CISIAV1NormalizedImgProvider(GroupedImgDataProvider):
    
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        return
    
    
    
    
    div = 3
    data_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/Output/CASIA-IrisV1/NormalizedImages/"
    width = 512
    height = 64
    channels = 3
    suffix = "_imno.bmp"