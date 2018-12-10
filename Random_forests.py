#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
Empirical good default values are max_features=n_features for regression problems, 
and max_features=sqrt(n_features) for classification tasks (where n_features is the
number of features in the data)
"""
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_process import FeatureLabel
from sklearn import preprocessing
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
import warnings
import sklearn.exceptions
import numpy as np
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

class RandomForests(object):
    def __init__(self,data,scale=True):
        self.logger = logging.getLogger("Bagging")
        if scale:
            self.X = preprocessing.scale(data[:,:-1])
        else:
            self.X = data[:, :-1]
        self.Y = data[:,-1]

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        a = accuracy_score(y_true,y_predict)
        return f,p,r,a
    #train_test_split
    def randomtree(self,n_estimators=None,max_features='auto',min_samples_split=2,name="RandomTree"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_features=max_features,
                                     min_samples_split = min_samples_split,
                                     random_state=0)
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def randomtree_cv(self,cv=5,n_estimators=None,max_features='auto',
                      min_samples_split=2,name="RandomTree_cv"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                     min_samples_split = min_samples_split,random_state=0)
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def randomtree_out(self,n_estimators=None,max_features='auto',
                       min_samples_split=2,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                         min_samples_split=min_samples_split,random_state=0)
            clf.fit(train_X, train_y)
            predict = clf.predict(test_x)
            truelabel.append(test_y)
            predictlabel.append(predict)

        f, p, r,a = self.nbeval(truelabel, predictlabel)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

if __name__=="__main__":

    FL = FeatureLabel()
    data = FL.features_label()
    SRT = RandomForests(data)
    #train_test_split
    SRT.randomtree(n_estimators=10)

    SRT.randomtree_cv(n_estimators=10)

    SRT.randomtree_out(30,"sqrt")

    print("Use data Scale!")

    BT = RandomForests(data,False)
    # train_test_split
    BT.randomtree(n_estimators=10)

    BT.randomtree_cv(n_estimators=10)

    BT.randomtree_out(30,"sqrt")


