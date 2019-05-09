import random

import numpy as np
import pandas as pd
from sklearn import svm


# SVM等价于线性规划问题，所以，线性规划问题分成两部分，目标函数和约束函数
# 对于目标函数的优化，通过采用SMO算法，进行迭代，可求出符合要求的可行解
def ker(x1, x2, type=0, sigma=1):
    # type表示核函数的类型，这里用0表示线性核，1表示高斯核()，其它类型的核函数待补充;
    global ker_val, df_input, gamma
    if gamma != 0:
        sigma = gamma
    temp = ker_val[x1, x2]
    if temp != 0:
        return temp
    else:
        nx1 = df_input[x1]
        nx2 = df_input[x2]
        if type == 0:
            kernel = np.dot(nx1, nx2)
            ker_val[x1, x2] = kernel
            return kernel
        elif type == 1:
            x_diff = np.subtract(nx1, nx2)
            tempk = np.dot(x_diff, x_diff) / (-2 * sigma ** 2)
            kernel = np.exp(tempk)
            ker_val[x1, x2] = kernel
            return kernel


# 在SMO算法的迭代中，E的计算需要经常用到，所以单独写一个关于计算E的函数(Ei=ui-yi)

def cal_E(i, type):
    global y, a, b, N
    u = 0
    for j in range(N):
        temp = y[j] * a[j] * ker(j, i, type)
        u += temp
    # print(u)
    u = u + b
    E_val = u - y[i]
    return E_val


def cal_LnH(i, j, y, a, C):
    if y[i] != y[j]:
        L = max(0, a[j] - a[i])
        H = min(C, C + a[j] - a[i])
    else:
        L = max(0, a[j] + a[i] - C)
        H = min(C, a[j] + a[i])
    # print(L,H)
    return L, H


# 一轮迭代所需要的计算过程，首先，计算a2的可能值；然后，通过L、H的限定，得到新的a2
def iteration_a2(j, type, toler):
    # print(j)
    global N, a, C, y, b, ker_val
    i = find_a1(j, type, toler)
    aj_old = a[j]
    ai_old = a[i]
    temp = y[j] * (cal_E(i, type) - cal_E(j, type)) / (ker(i, i, type) + ker(j, j, type) - ker(i, j, type))
    atemp = a[j] + temp

    L, H = cal_LnH(i, j, y, a, C)
    if atemp <= L + toler:
        a[j] = L
    elif (atemp >= H - toler):
        a[j] = H
    else:
        a[j] = atemp
    # 至此，新的a2计算完毕，下面计算新的a1
    ai_new = ai_old + (aj_old - a[j]) * y[i] * y[j]
    if abs(ai_new - ai_old) < toler:
        # print(abs(ai_new - ai_old))
        # print("too small")
        return 0
    else:
        a[i] = ai_new
    # 下面更新b
    b1 = -cal_E(j, type) - y[i] * (a[i] - ai_old) * ker(i, i, type) - y[j] * (a[j] - aj_old) * ker(i, j, type) + b
    b2 = -cal_E(j, type) - y[i] * (a[i] - ai_old) * ker(i, j, type) - y[j] * (a[j] - aj_old) * ker(j, j, type) + b
    if 0 < a[i] < C:
        b = b1
    elif 0 < a[j] < C:
        b = b2
    else:
        b = (b1 + b2) / 2.0
    return 1


## 对于固定下来的alpha2 如何取alpha1.遵循四条顺序原则：1）寻找最大的E的差值 2）不违反KKT条件
# 后采用随机取非alpha2，反而效果更好，为什么呢
def find_a1(j, type, toler=0.0001):
    global N, a, C
    # index = j
    # while (j == index):
    #     index = int(random.uniform(0, N))
    # return index
    index = -1
    diff = toler
    # print("123")
    for i in range(N):
        Ei = cal_E(i, type)
        ai = a[i]
        # print(Ei)
        # print(ai)
        if not ((ai < C and Ei < 0) or (ai > 0 and Ei > 0)):
            tempD = abs(Ei - cal_E(j, type))
            # print("i:"+str(i)+" di:" + str(tempD))
            if tempD > diff:
                diff = tempD
                index = i
    # print(index)
    return index


# 对于参数alpha的更新，其基本的原则遍历所有的a，取不符合KKT条件的a作为为第一个参数

def SMO(step, type, toler):
    count = 0
    i = 0
    while count < step:
        i = int(random.uniform(0, N))
        while iteration_a2(i, type, toler):
            count += 1
        count += 1
    # print(a)
    # print(b)


# 最后要求出w
def cal_W(alist):
    global y, df_input
    w = np.zeros(df_input.shape[1])
    for i in range(N):
        q = y[i] * alist[i] * df_input[i]
        w = np.add(w, q)
    return w


def main():
    df0 = pd.read_csv("data/iris.csv")

    df0.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    df1 = df0[df0['Species'] == 'setosa']
    df2 = df0[df0['Species'] == 'versicolor']

    df_se = df1[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[0:39]
    df_vi = df2[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[50:89]

    print(df_vi)

    df_se_test = df1[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[40:49]
    df_vi_test = df2[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[90:99]

    y_test = np.ones(20)
    for i in range(10):
        y_test[i + 10] = -1
    df_test = df_se_test.append(df_vi_test)

    # df_test = df_test.apply(lambda x: (x-np.min(x))/ (np.max(x)-np.min(x))).values
    df_test = df_test.apply(lambda x: (x - np.mean(x)) / np.std(x)).values

    df3 = df_se.append(df_vi)


    df_minmax = df3.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_zscore = df3.apply(lambda x: (x - np.mean(x)) / np.std(x))

    global N, a, y, ker_val, df_input, b, C, gamma

    df_input = df_zscore.values
    N = df_input.shape[0]
    a = np.zeros(N)
    y = np.ones(N)
    ker_val = np.mat(np.zeros([N, N]))
    gamma = 0.25
    C = 10
    b = 0.0

    for k in range(df_se.shape[0]):
        y[N - k - 1] = -1

    SMO(500, 0, 0.0001)
    w = cal_W(a)
    count = 0
    for q in range(20):
        if (np.dot(w, df_test[q]) + b) * y_test[q] > 0:
            count += 1
    print("mysvm accuracy:" + str(count / 20))

    model = svm.SVC(kernel="linear")
    model.fit(df_input, y)
    count = 0
    for q in range(20):
        if (model.predict([df_test[q]]) * y_test[q]) > 0:
            count += 1
    print("svm.SVC accuracy:" + str(count / 20))


if __name__ == '__main__':
    main()
