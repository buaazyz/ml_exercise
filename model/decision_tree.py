import numpy as np
import pandas as pd

def cal_Ent(train_set, attr):
    #此处的attr其实是标签项
    dic = {}
    for index, row in train_set.iterrows():
        if dic.get(row[attr]) != None:
            dic[row[attr]] = dic[row[attr]] + 1
        else:
            dic[row[attr]] = 1

    total = train_set.shape[0]
    sum = 0.0
    for key in dic:
        temp = dic[key]/total
        sum += temp * np.log2(temp)

    return sum


def Gain(train_set, attr, label):
    #此处的attr是属性
    E_exp = cal_Ent(train_set,label)
    Emap = {}
    # attr_list =[]
    attr_list = train_set[attr].apply(lambda x: x)
    attr_list = list(set(attr_list))
    total = train_set.shape[0]
    loss = 0.0
    for i in attr_list:
        temp = train_set[train_set[attr] == i]
        temp_num = temp.shape[0]
        loss += temp_num/total * cal_Ent(temp,label)
    gain = E_exp - loss
    return gain

# def iterator():
#
#
#
#
# def ser2dis(train_set, attr):
#
#
