#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import logging
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_process import FeatureLabel
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
import warnings
import sklearn.exceptions
import numpy as np
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

class SVMClassifier(object):
    def __init__(self,data):
        self.logger = logging.getLogger("SVM")
        self.X = preprocessing.scale(data[:,:-1])
        self.Y = data[:,-1]

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        a = accuracy_score(y_true,y_predict)
        return f,p,r,a
    #train_test_split
    #kernel=linear ovr
    def svc(self,name="Train_Test,ovr"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

    #svm one-against-one
    # kernel='linear'  ovo
    def svc_ovo(self,name="one-against-one"):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=0
        )
        clf = svm.SVC(kernel='linear',decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        f, p, r,a = self.nbeval(y_test, predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

    #svm one-vs-the-rest
    def svc_ovr(self,name="one-vs-the-rest"):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=0
        )
        clf = svm.LinearSVC(max_iter=6000)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        f, p, r,a = self.nbeval(y_test, predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

    #svm one-vs-rest cross
    def svc_ovr_cv(self,cv=5,name="OVR-CV"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = svm.LinearSVC(max_iter=900)
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),np.mean(r))
        self.logger.info(line)

    # For unbalanced problems
    #SVC (but not NuSVC) implement a keyword class_weight in the fit method.
    # It’s a dictionary of the form {class_label : value},
    # where value is a floating point number > 0 that sets the parameter
    # C of class class_label to C * value.
    def svc_cv_unbalance(self,cv=5,kernel='linear',gamma='auto_deprecated',name="Unbalance"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = svm.SVC(kernel=kernel,gamma=gamma,class_weight='balanced')
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),np.mean(r))
        self.logger.info(line)

    #leave one out
    def svm_one_out(self,kernel='linear',gamma='auto',name="Leave_one_out"):
        loo = LeaveOneOut()
        truelabel = []
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = svm.SVC(kernel=kernel,gamma=gamma,class_weight='balanced')
            clf.fit(train_X, train_y)
            predict = clf.predict(test_x)
            truelabel.append(test_y)
            predictlabel.append(predict)
        f, p, r, a = self.nbeval(truelabel, predictlabel)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f} Accuracy score:{:.3f}.".format(name, f, p, r,a)
        self.logger.info(line)

if __name__=="__main__":

    FL = FeatureLabel()
    data = FL.features_label()
    SVMC = SVMClassifier(data)
    #train_test_split
    SVMC.svc()

    SVMC.svc_ovo()
    #
    SVMC.svc_ovr()
    #
    SVMC.svc_ovr_cv()
    #
    SVMC.svc_cv_unbalance()

    SVMC.svc_cv_unbalance(5,'rbf','scale')

    SVMC.svm_one_out(gamma='scale')

    SVMC.svm_one_out('rbf')

