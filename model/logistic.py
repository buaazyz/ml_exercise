import numpy as np
import pandas as pd


def calculate(w, xi):
    lin = np.dot(w, xi)
    res = 1.0 / (1 + np.exp(lin))
    return res


def w_iter(w, xTrain, y, a=0.001, type='gd'):
    w_old = w
    # 其实不用一个一个维度的更新，因为维度之间是默认没有联系的
    if type == 'gd':
        delta = 0.0
        for i in range(xTrain.shape[0]):
            loss = calculate(w, xTrain[i]) - y[i]
            delta += xTrain[i] * loss
        w_new = w_old + a * delta / xTrain.shape[0]
    elif type == 'sgd':
        k = np.random.randint(0, xTrain.shape[0])
        loss = calculate(w, xTrain[k]) - y[k]
        delta = xTrain[k] * loss
        w_new = w_old + a * delta
    elif type == 'newton':
        print(1)

    return w_new


def main():
    df0 = pd.read_csv("data/iris.csv")

    df0.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    df1 = df0[df0['Species'] == 'setosa']
    df2 = df0[df0['Species'] == 'versicolor']

    df_se = df1[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[0:39]
    df_vi = df2[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[50:89]

    df_se_test = df1[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[40:49]
    df_vi_test = df2[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].loc[90:99]

    y_test = np.ones(20)
    for i in range(10):
        y_test[i + 10] = 0
    df_test = df_se_test.append(df_vi_test)


    df_test_MM = df_test.apply(lambda x: (x-np.min(x))/ (np.max(x)-np.min(x)))
    df_test_ZS = df_test.apply(lambda x: (x - np.mean(x)) / np.std(x))

    df_test_MM.insert(0, 'b', 1)
    df_test_ZS.insert(0, 'b', 1)

    df_test = df_test_MM.values

    df3 = df_se.append(df_vi)




    df_train_MM = df3.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_train_ZS = df3.apply(lambda x: (x - np.mean(x)) / np.std(x))

    # 添加偏置项
    df_train_MM.insert(0,'b',1)
    df_train_ZS.insert(0,'b',1)

    w = np.zeros(df_train_MM.shape[1])
    step = 1000
    a = 0.001

    df_input = df_train_MM.values

    y = np.ones(df_train_MM.shape[0])
    for k in range(df_se.shape[0]):
        y[df_train_MM.shape[0] - k - 1] = 0


    for i in range(step):
        w = w_iter(w, df_input, y,a, type='gd')


    print(w)
    count = 0
    for d in range(20):
        print(calculate(w,df_test[d]))
        if(calculate(w,df_test[d]) > 0.5 and y_test[d] == 1):
            count += 1
        elif(calculate(w,df_test[d]) < 0.5 and y_test[d] == 0):
            count += 1

    print("accuracy:",count/20)

if __name__ == '__main__':
    main()