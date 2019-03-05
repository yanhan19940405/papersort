import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import jieba
import math
import datetime
import time
from gensim import corpora
from gensim.summarization import bm25
def timeselect(timelist,selectvalue):#timelist:指新闻发布时间构成列表，selectvalue指设置的时间阈值，单位：天
    localtime = datetime.datetime.today()
    id_list=[]
    i=0
    for i in range(len(timelist)):
        time = (localtime - datetime.datetime.strptime(timelist[i], '%Y-%m-%d')).days
        if time<=selectvalue:
            id_list.append(i)
        else:
            pass
    i=i+1
    return id_list
def BMsort(list1,query_str):##list1指新闻构成的列表，query_str:指用户输入检索词
    dic = corpora.Dictionary(list1)
    bm25Model = bm25.BM25(list1)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    query_str = jieba.cut(query_str)
    query_str=" ".join(query_str)
    query = []
    for word in query_str.strip().split():
        query.append(word)
    scores = bm25Model.get_scores(query, average_idf)
    return scores
def labellist(label):#情感标签列表数值替换
    labelvalue=[]
    for i in label:
        if i=="pos":
            labelvalue.append(math.exp(1.0))
        elif i=="neu":
            labelvalue.append(math.exp(0.0))
        elif i=="neg":
            labelvalue.append(math.exp(-1.0))
    return labelvalue
def math_cau(x,y):
    list_W0=np.multiply(np.array(x), np.array(y))
    return  list_W0.tolist()

def by_score(t):##元组排序关键词
    return -t[1]
def selectshow(w,id_list):#新闻文本排序后输出
    i=0
    tup_list=[]
    for i in range(len(id_list)):
        tup=(id_list[i],w[i])
        tup_list.append(tup)
        i=i+1
    L3= sorted(tup_list,key = by_score)
    news_list=list(map(lambda x:(dframe["原文"].tolist())[x[0]],L3))
    return news_list
if __name__ == '__main__':
    time1=time.time()
    dframe = pd.read_excel('data.xlsx')
    new_time = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in dframe['时间']]
    id_list=timeselect(new_time,selectvalue=7)
    list1=[jieba.lcut(i) for i in dframe["原文"] if (dframe["原文"].tolist()).index(i) in id_list]
    print(list1)
    label=[i for i in dframe['情感预测']]
    label_1=[]
    for j in id_list:
        label_1.append(label[j])
    scores=BMsort(list1,query_str="山东能源集团")
    labelvalue=labellist(label_1)
    list_W0=math_cau(scores,labelvalue)
    print("输出新闻内容:",selectshow(list_W0, id_list))
    print("新闻条数",len(selectshow(list_W0, id_list)))
    time2 = time.time()
    print("耗时:",str(time2-time1)+"s")

