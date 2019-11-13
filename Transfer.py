#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:29:15 2019

@author: ninn
"""

class Transferer:
    origin_apath = ""
    target_apath = ""
    suffix = ".bmp"
        

    def _transfer(self, copy=True):
        import utils
        utils.mkapath(self.target_apath)
        import os,shutil
        dirs = [self.origin_apath]
        while len(dirs)>0 :
            tmpdir = dirs.pop()
            if os.path.isdir(tmpdir):
                tmpdirs = os.listdir(tmpdir)
                tmproot = tmpdir
                for tmpdir in tmpdirs:
                    dirs.append(os.path.join(tmproot, tmpdir))
            else:
                basename = os.path.basename(tmpdir)
                if basename.endswith(self.suffix):
                    origin_adir = os.path.join(self.origin_apath, tmpdir)
                    target_adir = os.path.join(self.target_apath, basename)
                    if copy:
                        self._copyto(origin_adir, target_adir)
                    else:
                        self._moveto(origin_adir, target_adir)
        return
    def _copyto(self, origin_adir, target_adir):
        import shutil
        print("[.]{} copy to {}".format(origin_adir, target_adir))
        try:
            shutil.copyfile(origin_adir, target_adir)
            print("[o]{} copy to {}".format(origin_adir, target_adir))
        except:
            print("[x]{} copy to {}".format(origin_adir, target_adir))
        return
    def _moveto(self, origin_adir, target_adir):
        import shutil
        print("[.]{} move to {]".format(origin_adir, target_adir))
        try:
            shutil.move(origin_adir, target_adir)
            print("[o]{} move to {]".format(origin_adir, target_adir))
        except() as e:
            print("[x]{} move to {]".format(origin_adir, target_adir))
            print(e)
        return
    def copy(self):
        self._transfer(True)
        return
    def move(self):
        self._transfer(self)
        self._transfer(False)
        return

class CASIAV1(Transferer):
    origin_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-IrisV1/"
    target_apath = "/home/ninn/bishe/iris/my/unet/data/CASIA-IrisV1/"

    

    def _transfer(self):
        import os
        assert os.path.isdir(self.origin_apath)
        
        import utils
        utils.mkapath(self.target_apath)
        
        
        

class CASIAV2Mover(Transferer):
    origin_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-IrisV2/"
    target_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-IrisV2/"

    def transfer(self):
        super().transfer(copy=False)
        return
    pass
    
class CASIAV4ThousandMover(Transferer):
    origin_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-Iris-Thousand/"
    target_apath = "/home/ninn/bishe/iris/git/Iris_Osiris/data/CASIA-Iris-Thousand/"
    suffix = ".jpg"
    def transfer(self):
        super().transfer(copy=False)
        return
    pass
if __name__ == '__main__':
    data = CASIAV4ThousandMover()
    data.transfer()
