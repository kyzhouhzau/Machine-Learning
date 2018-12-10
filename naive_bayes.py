#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import logging
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from data_process import FeatureLabel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
import warnings
from sklearn.model_selection import StratifiedKFold
import sklearn.exceptions
import numpy as np
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

class NaiveBayes(object):
    def __init__(self,data):
        self.logger = logging.getLogger("NB")
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        return f,p,r

    #train_test_split
    def naive_bayes(self,name="Train_Test"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.4,random_state=0
        )
        clf = ComplementNB()
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

    #cross-validated
    def naive_bayes_cv(self,cv=5,name="Cross-Validated"):
        clf = ComplementNB()
        f_score = np.mean(cross_val_score(clf,self.X,self.Y,cv=cv,n_jobs=8,scoring='f1_macro'))
        line = "{}: Fload {} F score:{:.3f}.".format(name,cv,f_score)
        self.logger.info(line)

    #Pipline and use standardScaler()
    def pipline_nb(self,cv=5,name = "Pipline_NB"):
        clf = make_pipeline(preprocessing.MinMaxScaler(),ComplementNB())
        f_score= np.mean(cross_val_score(clf,self.X,self.Y,cv=cv,scoring='f1_macro'))
        line = "{}: Flod {} F score:{:.3f}.".format(name,cv, f_score)
        self.logger.info(line)

    #cross_validate function its better then cross_val_score function
    #This function can also be used in pipline function.
    def naive_bayes_cv2(self,cv=5,name="Cross_Validate"):
        scoring = ["f1_macro","precision_macro","recall_macro"]
        clf = ComplementNB()
        score = cross_validate(clf,self.X,self.Y,cv=cv,scoring=scoring,return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name,cv, np.mean(f), np.mean(p), np.mean(r))
        self.logger.info(line)

    #Leave One Out
    def naive_bayes_loo(self,name="Leave One Out"):

        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex,testindex in loo.split(self.X,self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = ComplementNB()
            clf.fit(train_X,train_y)
            predict = clf.predict(test_x)
            truelabel.append(test_y)
            predictlabel.append(predict)

        f, p, r = self.nbeval(truelabel, predictlabel)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

    # Stratified k-fold
    def naive_bayes_stratified(self,name="StratifiedKFold"):
        skf = StratifiedKFold(n_splits=5)
        F=[];P=[];R=[]
        for trainindex,testindex in skf.split(self.X,self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx]for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = ComplementNB()
            clf.fit(train_X,train_y)
            predict = clf.predict(test_x)
            f,p,r = self.nbeval(test_y,predict)
            F.append(f)
            P.append(p)
            R.append(r)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, np.mean(F), np.mean(P), np.mean(R))
        self.logger.info(line)

if __name__=="__main__":

    FL = FeatureLabel()
    data = FL.features_label()
    NB = NaiveBayes(data)
    #train_test_split
    NB.naive_bayes()
    #cross_validated
    NB.naive_bayes_cv(5)
    #pipline
    NB.pipline_nb()
    #cross_validate
    NB.naive_bayes_cv2()
    #Leave One Out
    NB.naive_bayes_loo()
    #naive_bayes_stratified
    NB.naive_bayes_stratified()




