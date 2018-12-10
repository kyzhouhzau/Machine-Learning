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
from sklearn.linear_model import LogisticRegression
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

class LogisticRegressionModel(object):
    def __init__(self,data,scale=True):
        self.logger = logging.getLogger("Lasso")
        if scale:
            self.X = preprocessing.scale(data[:, :-1])
        else:
            self.X = data[:, :-1]
        self.Y = data[:, -1]

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        a = accuracy_score(y_true,y_predict)
        return f,p,r,a
    #train_test_split
    def logisticregression(self,solver ="lbfgs" ,multi_class="multinomial" ,name="Train_Test"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = LogisticRegression(random_state=0, solver=solver,multi_class=multi_class)
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def logisticregression_cv(self,cv=5,solver ="lbfgs" ,multi_class="multinomial" ,name="logistic_cv"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = LogisticRegression(random_state=0, solver=solver,multi_class=multi_class)
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def logisticregression_out(self,solver ="lbfgs" ,multi_class="multinomial" ,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = LogisticRegression(random_state=0, solver=solver,multi_class=multi_class)
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
    SLM = LogisticRegressionModel(data)
    #train_test_split
    SLM.logisticregression()

    SLM.logisticregression_cv()

    SLM.logisticregression_out()

    SLM.logisticregression_out("lbfgs")

    print("Use data not Scale!")

    LM = LogisticRegressionModel(data,scale=False)
    # train_test_split
    LM.logisticregression()

    LM.logisticregression_cv()

    LM.logisticregression_out()






