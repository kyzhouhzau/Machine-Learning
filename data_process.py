#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import glob
import os
import numpy as np

class FeatureLabel(object):
    def __init__(self):
        self.files = glob.glob("./data/BIO/*")
        self.tagfile = "./data/name_label.txt"
        self.labelfile = "./data/label_id.txt"

    def get_label(self,file):
        with open(file) as rf:
            dom  = rf.read().strip()
            dom = eval(dom)
            return dom

    def load_labe(self,file):
        label_id = {}
        with open(file) as rf:
            for line in rf:
                contents = line.strip().split()
                label_name = contents[0]
                id = contents[-1]
                label_id[label_name]=id
            return label_id

    def features_label(self):
        name_tag = self.get_label(self.tagfile)
        label_id = self.load_labe(self.labelfile)
        feature_label = np.zeros((len(self.files),len(label_id)+1))
        for i,file in enumerate(self.files):
            rf = open(file)
            name = os.path.basename(file)
            tag=None
            for line in rf:
                contents = line.strip().split('\t')
                tag = name_tag[name]
                if contents[-1].startswith("B"):
                    label = contents[-1].split('-')[-1]
                    if label in label_id.keys():
                        lid = label_id[label]
                        feature_label[i,int(lid)]+=1
            feature_label[i, -1] = int(tag)
            rf.close()
        return feature_label

if __name__=="__main__":

    fl = FeatureLabel()
    print(fl.tagfile)
    feature_label = fl.features_label()
    print(feature_label[1:50,:])
