import numpy as np
import pandas as pd
from sklearn import *
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import preprocess as pp
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.externals import joblib


# 做一个基于svm的bagging形成一个超平面，之后通过两次超平面的判断的投票选择
def svm_train(ts1, ts2, bat=10):
    w_avg = np.zeros(ts1.shape[1] - 1)
    b = 0
    for i in range(bat):
        y1 = np.ones(200)
        y2 = np.ones(200)
        y2[y2 == 1] = -1
        batch_y = np.concatenate([y1, y2])
        batch1 = ts1.sample(n=200).drop(['label'], axis=1)
        batch2 = ts2.sample(n=200).drop(['label'], axis=1)
        batch_x = pd.concat([batch1, batch2])
        clf_svc_cv = svm.SVC(kernel='linear', C=0.1)
        # score = cross_val_score(clf_svc_cv,batch_x,batch_y, cv=10)
        clf_svc_cv.fit(batch_x, batch_y)
        w = clf_svc_cv.coef_[0]
        w_avg += w
        b += clf_svc_cv.intercept_

    w_avg = w_avg / 10
    b = b / 10
    return w_avg, b


def svm_bagging(tSet0,tSet1,tSet2):
    w_avg1, b1 = svm_train(tSet0, tSet1)
    w_avg2, b2 = svm_train(tSet0, tSet2)
    w_avg3, b3 = svm_train(tSet1, tSet2)
    w_avg = [w_avg1, w_avg2, w_avg3]
    b = [b1, b2, b3]
    output = open('output/model_svm.pkl', 'wb')
    pickle.dump(w_avg, output)
    pickle.dump(b, output)
    output.close()
    # eval_svm(w_avg, b, pre)



def boosting(pre, type = 'xgboost'):
    if(type == 'xgboost'):
        clf = xgb.XGBClassifier(
            n_estimators=20,  # 迭代次数
            learning_rate=0.1,  # 步长
            max_depth=5,  # 树的最大深度
            min_child_weight=1,  # 决定最小叶子节点样本权重和
            subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
            colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
            objective='multi:softmax',  # 多分类
            num_class=3,
            nthread=4)
        clf.fit(pre.x_train_ori, pre.y_train, verbose=True)
        print(pre.x_train_ori.shape)
        joblib.dump(clf, "output/xgboost.m")
        # acc_test = clf.score(pre.x_test_ori, pre.y_test)
        # print(acc_test)

    elif(type == 'gbdt'):
        gbr = GradientBoostingClassifier(n_estimators=100,  learning_rate=0.1,max_depth=2, min_samples_split=2)#定义基本同上
        gbr.fit(pre.x_train_ori, pre.y_train)
        joblib.dump(gbr, "output/gbdt.m")
        # acc_test = gbr.score(pre.x_test_ori, pre.y_test)
        # print(acc_test)

    else:
        print("type error")


def main():
    pre = pp.preprocess()
    pre.getData()
    # pre.tsne_plot()

    x_train = pre.x_train
    y_train = pre.y_train

    # -------- A方案：对于label 2采用欠采样，对于label 0才用过采样到200数量，  -------- #

    # print(x_train.columns)
    # print(x_train.describe())
    # estimator = PCA(n_components=20)
    # X_pca = estimator.fit_transform(x_train.drop(['ID']))

    trainSet = pd.concat([x_train, y_train], axis=1)

    trainSet0 = trainSet[trainSet['label'] == 0]
    trainSet1 = trainSet[trainSet['label'] == 1]
    trainSet2 = trainSet[trainSet['label'] == 2]

    trainSet_over = trainSet[trainSet['label'].isin([0, 2])]
    y = trainSet_over['label'].values
    y[y == 0] = 1
    y[y == 2] = -1

    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(trainSet_over, y)
    X_smo = pd.DataFrame(X_smo)
    # print(type(X_smo))
    X_smo.columns = trainSet1.columns
    trainSet0 = X_smo[X_smo['label'] == -1]

    # svm_bagging(trainSet0, trainSet1, trainSet2)
    boosting(pre,type='xgboost')


if __name__ == '__main__':
  main()
