#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import logging
import autosklearn.classification
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

class AutoSklearn(object):
    def __init__(self,data,scale=True):
        self.logger = logging.getLogger("AutoSklearn")
        if scale:
            self.X = preprocessing.scale(data[:, :-1])
        else:
            self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.clf = autosklearn.classification.AutoSklearnClassifier(
            include_estimators=["random_forest", "libsvm_svc", "k_nearest_neighbors"],
            include_preprocessors=["no_preprocessing", ],
            ensemble_size=1,
            initial_configurations_via_metalearning=0
        )

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        a = accuracy_score(y_true,y_predict)
        return f,p,r,a
    #train_test_split
    def autosklearn(self ,name="Train_Test"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = self.clf
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def autosklearn_cv(self,cv=5,name="autosklearn_cv"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = self.clf
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def autosklearn_out(self,name="Leave One Out"):
        loo = LeaveOneOut()
        F = [];P = [];R = []
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = self.clf
            clf.fit(train_X, train_y)
            predict = clf.predict(test_x)
            f, p, r,a = self.nbeval(test_y, predict)
            F.append(f)
            P.append(p)
            R.append(r)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, np.mean(F), np.mean(P), np.mean(R))
        self.logger.info(line)

if __name__=="__main__":

    FL = FeatureLabel()
    data = FL.features_label()
    SAS = AutoSklearn(data)
    #train_test_split
    SAS.autosklearn()

    SAS.autosklearn_cv()

    SAS.autosklearn_out()
    print("Use data not Scale!")
    AS = AutoSklearn(data,scale=False)
    # train_test_split
    AS.autosklearn()

    AS.autosklearn_cv()

    AS.autosklearn_out()






