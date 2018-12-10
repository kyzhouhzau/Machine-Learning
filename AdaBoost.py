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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from data_process import FeatureLabel
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
import warnings
import sklearn.exceptions
import numpy as np
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

class AdaBoost(object):
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
    def adaboost(self,n_estimators=None,name="RandomTree"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = AdaBoostClassifier(n_estimators=n_estimators,random_state=0)
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def adaboost_cv(self,cv=5,n_estimators=None,name="RandomTree_cv"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = AdaBoostClassifier(n_estimators=n_estimators,random_state=0)
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def adaboost_out(self,n_estimators=None,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        dt = DecisionTreeClassifier(max_depth=3)
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = AdaBoostClassifier(base_estimator=dt,n_estimators=n_estimators,random_state=0)
            clf.fit(train_X, train_y)
            predict = clf.predict(test_x)
            truelabel.append(test_y)
            predictlabel.append(predict)
        f, p, r, a = self.nbeval(truelabel, predictlabel)

        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)

if __name__=="__main__":

    FL = FeatureLabel()
    data = FL.features_label()
    SAB = AdaBoost(data)
    #train_test_split
    SAB.adaboost(n_estimators=100)

    SAB.adaboost_cv(n_estimators=100)

    SAB.adaboost_out(100)

    print("Use data Scale!")

    AB = AdaBoost(data,False)
    # train_test_split
    AB.adaboost(n_estimators=100)

    AB.adaboost_cv(n_estimators=100)

    AB.adaboost_out(100)


