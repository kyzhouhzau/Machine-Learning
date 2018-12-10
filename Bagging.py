#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
advantages
1、Simple to understand and to interpret. Trees can be visualised.
2、Requires little data preparation. Other techniques often require 
data normalisation, dummy variables need to be created and blank values 
to be removed. Note however that this module does not support missing values.
disadvantages 
1、Decision-tree learners can create over-complex trees that do not generalise the data well. 
2、Decision trees can be unstable because small variations in the data might result in a completely 
different tree being generated. This problem is mitigated by using decision trees within an ensemble.
3、

"""
import logging
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
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

class Baggingmethod(object):
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
    def baggingtree(self,name="BaggingTree"):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,self.Y,test_size=0.3,random_state=0
        )
        clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state=0))
        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        f,p,r,a = self.nbeval(y_test,predict)
        line = "{}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, f, p, r)
        self.logger.info(line)
    #cross_validate
    def baggingtree_cv(self,cv=5,max_depth=None,name="Decision_Tree"):
        scoring = ["f1_macro", "precision_macro", "recall_macro"]
        clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state=0,max_depth=max_depth))
        score = cross_validate(clf, self.X, self.Y, cv=cv, scoring=scoring, return_train_score=False)
        f = score["test_f1_macro"]
        p = score["test_precision_macro"]
        r = score["test_recall_macro"]
        line = "{}: Flod {}: F score:{:.3f}\tP score:{:.3f}\tR score:{:.3f}.".format(name, cv, np.mean(f), np.mean(p),
                                                                                     np.mean(r))
        self.logger.info(line)

    ##Leave One Out
    def baggingtree_out(self,max_depth=None,min_samples_leaf=1,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state=0,min_samples_leaf=min_samples_leaf,max_depth=max_depth))
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
    SBT = Baggingmethod(data)
    #train_test_split
    SBT.baggingtree()

    SBT.baggingtree_cv()

    SBT.baggingtree_cv(max_depth=4)

    SBT.baggingtree_out()

    SBT.baggingtree_out(max_depth=2)

    print("Use data Scale!")

    BT = Baggingmethod(data,False)
    # train_test_split
    BT.baggingtree()

    BT.baggingtree_cv()

    BT.baggingtree_cv(max_depth=4)

    BT.baggingtree_out()

    BT.baggingtree_out(max_depth=2)


