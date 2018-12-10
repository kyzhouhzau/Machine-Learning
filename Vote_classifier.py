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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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

class VoteClassifier(object):
    def __init__(self,data,scale=True):
        self.logger = logging.getLogger("VoteClassifier")
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
    def voteclassfier(self,n_estimators=None,voting="soft",max_depth=4,name="Voteclassfier"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf1 = LogisticRegression(solver='lbfgs',
                                  multi_class='multinomial',
                                  random_state = 1)
        clf2 = RandomForestClassifier(n_estimators=n_estimators,
                                     max_features="sqrt",
                                     random_state=1)
        clf3 = SVC(gamma='scale',class_weight='balanced',probability=True)
        clf4 = DecisionTreeClassifier(random_state=0,max_depth=max_depth)
        clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3),('tree',clf4)], voting=voting)

        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def voteclassfier_cv(self,cv=5,n_estimators=None,voting="soft",max_depth=4,name="Voteclassfier_cv"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf1 = LogisticRegression(solver='lbfgs',
                                  multi_class='multinomial',
                                  random_state=1)
        clf2 = RandomForestClassifier(n_estimators=n_estimators,
                                      max_features="sqrt",
                                      random_state=1)
        clf3 = SVC(gamma='scale',class_weight='balanced',probability=True)
        clf4 = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
        clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3), ('tree', clf4)], voting=voting)

        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def voteclassfier_out(self,n_estimators=None,voting="soft",max_depth=4,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf1 = LogisticRegression(solver='lbfgs',
                                      multi_class='multinomial',
                                      random_state=1)
            clf2 = RandomForestClassifier(n_estimators=n_estimators,
                                          max_features="sqrt",
                                          random_state=1)
            clf3 = SVC(gamma='scale',class_weight='balanced',probability=True)
            clf4 = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
            clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3),('tree',clf4)],
                                   voting=voting)
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
    SVOC = VoteClassifier(data)
    #train_test_split
    SVOC.voteclassfier(n_estimators=10)

    SVOC.voteclassfier_cv(n_estimators=10)

    SVOC.voteclassfier_out(30)

    print("Use data Scale!")

    VC = VoteClassifier(data,False)
    # train_test_split
    VC.voteclassfier(10)

    VC.voteclassfier_cv(n_estimators=10)

    VC.voteclassfier_out(30)


