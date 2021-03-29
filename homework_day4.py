# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""


import pandas as pd
import numpy as np
dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)
dataset.head(5)

dataset = dataset.fillna('')
dataset.head(5)

dataset_combined = dataset[0]
for i in range(1, len(dataset.columns)):
    dataset_combined = dataset_combined + ',' + dataset[i]
dataset_combined = pd.DataFrame(dataset_combined)
dataset_combined.columns = ['transactions']
dataset_combined.head(10)

dataset = dataset_combined['transactions'].str.get_dummies(',')
dataset.head(5)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(dataset, min_support=0.015, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support" , ascending=False)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
rules = rules.sort_values(by="lift" , ascending=False)
print("频繁项集：", frequent_itemsets)
print("关联规则：", rules)

from efficient_apriori import apriori

dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None) 
# 将数据存放到transactions中
transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0, 20):
        if str(dataset.values[i, j]) != 'nan':
           temp.append(str(dataset.values[i, j]))
    transactions.append(temp)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.015,  min_confidence=0.2)
print("频繁项集：", itemsets)
print("关联规则：", rules)

    