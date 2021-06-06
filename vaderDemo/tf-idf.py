from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import docx
import numpy as np
import csv
from collections import defaultdict
import math
import operator
import re

document = docx.Document("testdata/Dataset.docx")
sentences=[]
for para in document.paragraphs:
    sentences.append(para.text)

# print(sentences)
doc_frequency = defaultdict(int)
list=[]
for sentence in sentences:
    list.append(re.split(r'[.,;!?"\n\' ]', sentence))

words=[]
for i in list:
    for j in i:
        if j != '':
            words.append(j.lower())

dict = []
for word in words:
    if word not in dict:
        dict.append(word)

words_dic={}

iter=0
for word in dict:
    words_dic[word.lower()]=iter
    iter+=1

print(words_dic)
# print(len(dict))
# print(len(words))
print(words_dic['last'])

for word in words:
    doc_frequency[word] += 1

# print(doc_frequency)
# 计算每个词的TF值
word_tf = {}  # 存储没个词的tf值
for i in doc_frequency:
    word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

# 计算每个词的IDF值
doc_num = len(list)
word_idf = {}  # 存储每个词的idf值
word_doc = defaultdict(int)  # 存储包含该词的文档数
for i in doc_frequency:
    for j in list:
        has=0
        for word in j:
            if word.lower()==i:
                has=1
                break
        if has:
            word_doc[i] += 1

for i in doc_frequency:
    word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

# 计算每个词的TF*IDF的值
word_tf_idf = {}
for i in doc_frequency:
    word_tf_idf[i] = word_tf[i] * word_idf[i]

# 对字典按值由大到小排序
# dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)

print(word_idf)
print(word_tf_idf)



