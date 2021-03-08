# 使用KMeans进行聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

#data = pd.read_csv('Mall_Customers.csv', encoding='gbk')
data = pd.read_csv('car_data.csv',encoding='gbk')
print(data) #打印数据
train_x = data[["人均GDP","城镇人口比重","交通工具消费价格指数", "百户拥有汽车量"]]


min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
print(train_x)


kmeans = KMeans(n_clusters=6)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)

result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(result)

result.to_csv("customer_cluster_result.csv",index=False)


import matplotlib.pyplot as plt
sse = []
for k in range(1, 11):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('K')
plt.ylabel('sse')
plt.plot(x, sse, 'o-')
plt.show()

from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
model = AgglomerativeClustering(linkage='ward', n_clusters=3)
y = model.fit_predict(train_x)
print(y)

linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()