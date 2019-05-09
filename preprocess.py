import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

class preprocess:
    x_train = None
    y_train = None
    train_num = None
    test_num = None
    x_test = None
    y_test = None
    x_train_ori = None

    def __init__(self):
        pd.set_option('display.width', 100)  # 设置字符显示宽度
        pd.set_option('display.max_rows', None)  # 设置显示最大行
        pd.set_option('display.max_columns', None)  # 设置最大列数

    def getData(self):
        df1 = pd.read_csv('data/cone.csv')
        df2 = pd.read_csv('data/uncertain.csv')
        df3 = pd.read_csv('data/normal.csv')

        # 删除读取的时候无效的行和列，该行或列的全部值为空
        df1.dropna(how='all', axis=1, inplace=True)
        df2.dropna(how='all', axis=1, inplace=True)
        df3.dropna(how='all', axis=1, inplace=True)

        df1.dropna(how='all', inplace=True)
        df2.dropna(how='all', inplace=True)
        df3.dropna(how='all', inplace=True)

        # df3的后四列为无效的数据，故删去
        df3.drop(['Unnamed: 162', 'Unnamed: 163', 'Unnamed: 164', 'Unnamed: 165'], axis=1, inplace=True)

        df1.insert(162, 'label', 0)
        df2.insert(162, 'label', 1)
        df3.insert(162, 'label', 2)

        df0 = df1.append(df2.append(df3))
        # print(df0.shape)

        df0['Eye'].replace({'Left': 0, 'Right': 1}, inplace=True)
        df0['Eye'].replace({' Left': 0, ' Right': 1}, inplace=True)

        # 处理关于值为None的和数值中带有?的数据

        column = df0.columns
        types = df0.dtypes

        editcol = []

        for c in column:
            if (types[c] == 'object'):
                editcol.append(c)

        for index, row in df0.iterrows():
            for ec in editcol:
                temp = str(row[ec])
                # 经尝试，只有英文字符的?，没有中文字符的？
                if temp.find('?') != -1:
                    num = temp.replace('?', '.')
                    # 发现替换完后有重复的 . 所以用正则表达式
                    matchObj = re.search(r'\d+\.[^\.]*', num)
                    df0.loc[index, ec] = matchObj.group()
                if temp == ' None':
                    df0.loc[index, ec] = 'Nan'

        df0 = df0.astype(np.float64)

        # 该两行为查看每一列的缺失值的数目
        # for ec in editcol:
        # print(df0[ec].isnull().value_counts())

        # 考虑到PTI 8mm的缺失值过多,PTI 0mm的值全部为0，故删去该列,其他列用平均值代替缺失值

        for ec in editcol:
            df0[ec].fillna(df0[ec].mean(), inplace=True)

        df0.drop(['PTI 8mm', 'BirthD', 'medicalD', 'PTI 0mm'], axis=1, inplace=True)

        df1 = df0[df0['label'] == 0]
        df2 = df0[df0['label'] == 1]
        df3 = df0[df0['label'] == 2]

        df1_test = df1.sample(frac=0.1, random_state=1)
        df1_train = df1[~df1['ID'].isin(df1_test['ID'])]
        df2_test = df2.sample(frac=0.1, random_state=1)
        df2_train = df2[~df2['ID'].isin(df2_test['ID'])]
        df3_test = df3.sample(frac=0.1, random_state=1)
        df3_train = df3[~df3['ID'].isin(df3_test['ID'])]

        train_num = []
        test_num = []
        train_num.append(df1_train.shape[0])
        train_num.append(df2_train.shape[0])
        train_num.append(df3_train.shape[0])
        test_num.append(df1_test.shape[0])
        test_num.append(df2_test.shape[0])
        test_num.append(df3_test.shape[0])

        df0_train = df1_train.append(df2_train.append(df3_train))
        df0_test = df1_test.append(df2_test.append(df3_test))


        droplist = ['ID']
        for i in df0_train.columns:
            if (df0_train[i].var() < 0.001):
                droplist.append(i)
        # print(droplist)
        df0_train = df0_train.drop(droplist,axis = 1)
        df0_test = df0_test.drop(droplist,axis=1)



        # lasso = Lasso(alpha=0.01)
        # lasso.fit(df0_train, df0_train['label'])
        # print( lasso.coef_)


        # 基于Tree - based feature selection采用Random Forests
        rf = RandomForestRegressor(random_state=12)
        rf.fit(df0_train.drop(['label'],axis=1), df0_train['label'])
        im_feat = rf.feature_importances_
        droplist = []
        col = df0_train.columns
        for i in range(len(im_feat)):
            if(im_feat[i] < 0.001):
                droplist.append(col[i])
        df0_train = df0_train.drop(droplist, axis=1)
        df0_test = df0_test.drop(droplist, axis=1)

        # print(df0_train.describe())
        # print(df0_train_zscore.describe())

        # 经检查，年龄为0岁的只有1例，占比小，所以认为是代表刚出生的孩子
        # print(df0['age'].value_counts())
        # corr_matrix = df0.corr()

        self.x_train_ori = df0_train
        self.x_test_ori = df0_test


        # 对训练集标准化处理
        df0_train_zscore = df0_train.apply(lambda x: (x - np.mean(x)) / np.std(x))



        df0_train_zscore_x = df0_train_zscore.drop(['label'], axis=1)
        self.x_train = df0_train_zscore_x
        self.y_train = df0_train['label']
        self.train_num = train_num
        self.test_num = test_num
        self.x_test = df0_test.drop(['label'], axis=1)
        self.y_test = df0_test['label']
        # return df0_train_zscore_x, y_digits, train_num, test_num

    def pca_3D(self):
        estimator = PCA(n_components=3)
        # self.x_train.drop(['ID'], axis=1)
        X_pca = estimator.fit_transform(self.x_train)

        fig = plt.figure()
        ax = Axes3D(fig)
        colors = ['blue', 'green', 'yellow']
        for i in range(len(colors)):
            px = X_pca[:, 0][self.y_train.as_matrix() == i]
            py = X_pca[:, 1][self.y_train.as_matrix() == i]
            pz = X_pca[:, 2][self.y_train.as_matrix() == i]
            ax.scatter(px, py, pz, c=colors[i])
        # plt.legend(np.arange(0, 10).astype(str))
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.show()

    def tsne_plot(self):
        self.x_train_ori.drop(['ID'], axis=1)
        X_tsne = TSNE(n_components=3, learning_rate=100).fit_transform(self.x_train)
        # plt.figure(figsize=(12, 6))
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y_train)
        # plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        colors = ['blue', 'green', 'yellow']
        for i in range(len(colors)):
            px = X_tsne[:, 0][self.y_train.as_matrix() == i]
            py = X_tsne[:, 1][self.y_train.as_matrix() == i]
            pz = X_tsne[:, 2][self.y_train.as_matrix() == i]
            ax.scatter(px, py, pz, c=colors[i])
        # plt.legend(np.arange(0, 10).astype(str))
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.show()