import pickle
import numpy as np
from sklearn.externals import joblib
import preprocess as pp
from sklearn import metrics

pre = pp.preprocess()
pre.getData()


def eval_svm(w_avg, b, pre):
    x_test = pre.x_test
    x_test = x_test.apply(lambda x: (x - np.mean(x)) / np.std(x))
    y_test = pre.y_test.values
    count = 0
    i = 0
    for index, row in x_test.iterrows():
        label = -1
        judge = []
        for j in range(3):
            judge.append(np.dot(w_avg[j], row) + b[j])
        if judge[0] > 0 and judge[2] > 0:
            label = 0.0
        elif judge[0] < 0 and judge[1] > 0:
            label = 1.0
        elif judge[1] < 0 and judge[2] < 0:
            label = 2.0
        else:
            if judge[0] > 0:
                judge[2] = -judge[2]
                label = judge.index(max(judge))
            else:
                judge[0] = -judge[0]
                judge[1] = -judge[1]
                label = judge.index(max(judge))
                label = (label + 1) % 3
        if label == y_test[i]:
            count += 1
        # else:
        #     print("pre:" + str(label)+"    real:"+str(y_test[i]))
        i += 1
    print(count / x_test.shape[0])


def eval_boost(pre, type='xgboost'):
    if type == 'xgboost':
        clf = joblib.load("output/xgboost.m")
        acc_test = clf.score(pre.x_test_ori, pre.y_test)
        y_predict = clf.predict(pre.x_test_ori)
        print(acc_test)
    elif type == 'gbdt':
        # y_gbr1 = gbr.predict(x_test)
        gbr = joblib.load("output/gbdt.m")
        acc_test = gbr.score(pre.x_test_ori, pre.y_test)
        y_predict = gbr.predict(pre.x_test_ori)
        print(acc_test)
    y_test = pre.y_test
    print('宏平均精确率:', metrics.precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(y_test, y_predict, average='micro'))  # 预测微平均精确率输出
    print('加权平均精确率:', metrics.precision_score(y_test, y_predict, average='weighted'))  # 预测加权平均精确率输出

    print('宏平均召回率:', metrics.recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出
    print('微平均召回率:', metrics.recall_score(y_test, y_predict, average='micro'))  # 预测微平均召回率输出
    print('加权平均召回率:', metrics.recall_score(y_test, y_predict, average='micro'))  # 预测加权平均召回率输出

    print('宏平均F1-score:',
          metrics.f1_score(y_test, y_predict, labels=[0, 1, 2], average='macro'))  # 预测宏平均f1-score输出
    print('微平均F1-score:',
          metrics.f1_score(y_test, y_predict, labels=[0, 1, 2], average='micro'))  # 预测微平均f1-score输出
    print('加权平均F1-score:',
          metrics.f1_score(y_test, y_predict, labels=[0, 1, 2], average='weighted'))  # 预测加权平均f1-score输出

    print('混淆矩阵输出:\n', metrics.confusion_matrix(y_test, y_predict))  # 混淆矩阵输出
    print('分类报告:\n', metrics.classification_report(y_test, y_predict))  # 分类报告输出



def main():
    # pkl_file = open('output/model_svm.pkl', 'rb')
    # w_avg = pickle.load(pkl_file)
    # b = pickle.load(pkl_file)
    # print(b)
    # pkl_file.close()
    # eval_svm(w_avg, b, pre)

    eval_boost(pre, type='xgboost')


if __name__ == '__main__':
  main()

