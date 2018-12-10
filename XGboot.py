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
from xgboost import XGBClassifier
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

class XGBoot(object):
    def __init__(self,data,scale=True):
        self.logger = logging.getLogger("XGBoot")
        if scale:
            self.X = preprocessing.scale(data[:, :-1])
        else:
            self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.clf = XGBClassifier(learning_rate=0.001,
                            n_estimators=50,         # 树的个数--100棵树建立xgboost
                            max_depth=4,               # 树的深度
                            min_child_weight = 1,      # 叶子节点最小权重
                            gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                            subsample=0.8,             # 随机选择80%样本建立决策树
                            colsample_btree=0.8,       # 随机选择80%特征建立决策树
                            objective='multi:softmax', # 指定损失函数
                            scale_pos_weight=1,        # 解决样本个数不平衡的问题
                            random_state=27            # 随机数
                        )

    def nbeval(self,y_true,y_predict):
        f = f1_score(y_true,y_predict,average="macro")
        p = precision_score(y_true,y_predict,average="macro")
        r = recall_score(y_true,y_predict,average="macro")
        a = accuracy_score(y_true,y_predict)
        return f,p,r,a
    #train_test_split
    def xgboot(self,name="Train_Test"):
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
    def xgboot_cv(self,cv=5 ,name="xgboot_cv"):
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
    def xgboot_out(self,name="Leave One Out"):
        loo = LeaveOneOut()
        truelabel=[]
        predictlabel=[]
        for trainindex, testindex in loo.split(self.X, self.Y):
            train_X = np.array([self.X[idx] for idx in trainindex])
            train_y = np.array([self.Y[idx] for idx in trainindex])
            test_x = np.array([self.X[idx] for idx in testindex])
            test_y = np.array([self.Y[idx] for idx in testindex])
            clf = self.clf
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
    SXG = XGBoot(data)
    #train_test_split
    SXG.xgboot()

    SXG.xgboot_cv()

    SXG.xgboot_out()

    print("Use data not Scale!")

    XG = XGBoot(data,scale=False)
    # train_test_split
    XG.xgboot()

    XG.xgboot_cv()

    XG.xgboot_out()






