#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:48:59 2020

@author: zhangyingji
"""

import subprocess
import pyspark
from pyspark.sql import SparkSession
import re
import numpy as np
from pyspark.sql.functions import regexp_extract
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
import pyspark.sql.functions as func
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment1") \
    .getOrCreate()

sc = spark.sparkContext


movieFile = spark.read.csv("Data/ratings.csv", header=True).cache()
mf = movieFile.select(movieFile['userId'].cast('int'), movieFile['movieId'].cast('int'), movieFile['rating'].cast('float'), movieFile['timestamp'].cast('int'))


score = spark.read.csv("Data/genome-scores.csv", header=True).cache()
tag = spark.read.csv("Data/genome-tags.csv", header=True).cache()

def cross_valid(train, test):
    rmse, mae = [], []
    for reg in [0.01, 0.1, 0.5]:
        als = ALS(maxIter=5, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(train) # model.itemFactors
        predictions = model.transform(test)
        evaluator_r = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        evaluator_m = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
        r, m = evaluator_r.evaluate(predictions), evaluator_m.evaluate(predictions)
        rmse.append(r)
        mae.append(r)
    return rmse, mae, model.itemFactors
    
# -----------------------------------------------------------------------------
# 1. k-means.
# 2. get top3 cluster by counting the amount of movies.
# 3. get all movie ids of every cluster.
# 4. score.csv, get ranked tag id by sum score.
# 5. tag.csv, get top3 tag name by tag id.
# -----------------------------------------------------------------------------

def find_cluster_tag(dfVector):
    tag_list = []
    tag_num_list = []
    kmeans = KMeans(k=25, seed=1).setFeaturesCol("features")
    model = kmeans.fit(dfVector)
    movieId_cluster = model.transform(dfVector).select('id', 'prediction')
    # count the number of movies of every cluster.
    movieId_group = movieId_cluster.groupBy("prediction").count()
    # sort.
    cluster_count = movieId_group.orderBy(-movieId_group['count'])
    # collect top 3 cluster id.
    top3_cluster = cluster_count.take(3)
    top3_list = [i[0] for i in top3_cluster]
    # get movie ids that are in top3 clusters.
    for i in range(3):
        top_movieId_cluster = movieId_cluster.filter(movieId_cluster['prediction'] == top3_list[i])
        # get movie id of the first cluster.
        link = score.join(top_movieId_cluster, score.movieId == top_movieId_cluster.id, "inner")
        # used for counting the number of movie in a tag in a cluster.
        tagId_link = link.groupBy('tagId').count()
        # sum score.
        link1 = link.groupBy('tagId').agg(func.sum('relevance').alias('scores_sum'))
        # sort according to sum score.
        link2 = link1.orderBy(-link1['scores_sum'])
        # link tag.csv.
        top3_tag = link2.join(tag, link2.tagId == tag.tagId, "inner").take(3)
        tag_1 = [i[3] for i in top3_tag]
        tag_list.append(tag_1)
        # get top 3 tag id.
        tag_2 = [i[0] for i in top3_tag]
        for i in tag_2:
            tag_num = tagId_link.filter(tagId_link.tagId == i).take(1)
            tag_num_list.append(tag_num[0][1])
    return tag_list, tag_num_list

# ------------------------3 fold cross validation-------------------------------
data_1, data_2, data_3 = mf.randomSplit([0.3, 0.3, 0.4])

rmse_list = list()
mae_list = list()

# split 1.
training = data_1.union(data_3)
test = data_2
r, m, f = cross_valid(training, test)
# change arraytype to vector.
Vector = f.rdd.map(lambda r: (r['id'],) + tuple([Vectors.dense(r['features'])]))
dfVector = Vector.toDF(['id','features'])
tag_split_1, tag_num_1 = find_cluster_tag(dfVector)

rmse_list.append(r)
mae_list.append(m)

# split 2.
training = data_1.union(data_2)
test = data_3
r, m, f = cross_valid(training, test)
# change arraytype to vector.
Vector = f.rdd.map(lambda r: (r['id'],) + tuple([Vectors.dense(r['features'])]))
dfVector = Vector.toDF(['id','features'])
tag_split_2, tag_num_2 = find_cluster_tag(dfVector)

rmse_list.append(r)
mae_list.append(m)

# split 3.
training = data_2.union(data_3)
test = data_1
r, m, f = cross_valid(training, test)
# change arraytype to vector.
Vector = f.rdd.map(lambda r: (r['id'],) + tuple([Vectors.dense(r['features'])]))
dfVector = Vector.toDF(['id','features'])
tag_split_3, tag_num_3 = find_cluster_tag(dfVector)

rmse_list.append(r)
mae_list.append(m)
# -------------------------------------------------------

rmse_mean = np.array(rmse_list).mean(axis=0)
rmse_std = np.array(rmse_list).std(axis=0)

mae_mean = np.array(mae_list).mean(axis=0)
mae_std = np.array(mae_list).std(axis=0)

plt.figure(figsize=(8, 8))
# mean.
plt.bar([1, 3, 5], rmse_mean, width=0.5, label="rmse_mean")
plt.bar([1.5, 3.5, 5.5], mae_mean, width=0.5, label="mae_mean")
# std.
plt.bar([1, 3, 5], rmse_std*100, width=0.5, label="rmse std")
plt.bar([1.5, 3.5, 5.5], mae_std*100, width=0.5, label="mae std")

i, j = 0, 0
for index, x in enumerate([1, 1.5, 3, 3.5, 5, 5.5]):
    if index%2 == 0:
        plt.text(x, rmse_mean[i] + 0.01, '%.4f' % rmse_mean[i], ha="center", va="bottom")
        plt.text(x, rmse_std[i] + 0.03, '%.4f' % rmse_std[i], ha="center", va="bottom")
        i+=1
    else:
        plt.text(x, mae_mean[j] + 0.01, '%.4f' % mae_mean[j], ha="center", va="bottom")
        plt.text(x, mae_std[j] + 0.03, '%.4f' % mae_std[j], ha="center", va="bottom")
        j+=1

    
plt.xticks([1.2, 3.2, 5.2], ['0.01', '0.1', '0.5'])
plt.legend(loc="upper left")
plt.savefig('Q2_B')

# record Q2_A.

filename = 'Q2_output.txt'
with open(filename, 'a') as file_object:
    file_object.write("---------------------------Q2_A------------------------------ \n")
    file_object.write("-- RMSE regularisation strength: 0.01 0.1 0.5 --------------- \n")
    for i, rmse in enumerate(rmse_list):
        file_object.write("----------------------fold: "+str(i+1)+"-------------------------------- \n")
        file_object.write(str(rmse)+str("\n"))
    file_object.write("-------------------- RMSE mean ------------------------------ \n")
    file_object.write(str(rmse_mean)+str("\n"))
    file_object.write("-------------------- RMSE std ------------------------------- \n")
    file_object.write(str(rmse_std)+str("\n"))
    file_object.write("\n")
    file_object.write("--- MAE regularisation strength: 0.01 0.1 0.5 ---------------- \n")
    for i, mae in enumerate(mae_list):
        file_object.write("----------------------fold: "+str(i+1)+"--------------------------------- \n")
        file_object.write(str(mae)+str("\n"))

    file_object.write("-------------------- MAE mean -------------------------------- \n")
    file_object.write(str(mae_mean)+str("\n"))
    file_object.write("-------------------- MAE std --------------------------------- \n")
    file_object.write(str(mae_std)+str("\n"))
        
    file_object.write("-------------------------------------------------------------- \n")
    
with open(filename, 'a') as file_object:
    file_object.write("\n")
    file_object.write("------------------------------- Q2_C ----------------------------------- \n")
    file_object.write("------------------------ Top3 cluster tag ------------------------------ \n")
    file_object.write("----------------------- fold: 1 ---------------------------------------- \n")
    for i in tag_split_1:
        file_object.write(str(i)+str("\n"))
    file_object.write(str(tag_num_1)+str("\n"))
    file_object.write("----------------------- fold: 2 ---------------------------------------- \n")
    for i in tag_split_2:
        file_object.write(str(i)+str("\n"))
    file_object.write(str(tag_num_2)+str("\n"))
    file_object.write("----------------------- fold: 3 ---------------------------------------- \n")
    for i in tag_split_3:
        file_object.write(str(i)+str("\n"))
    file_object.write(str(tag_num_3)+str("\n"))
    file_object.write("------------------------------------------------------------------------ \n")
    
spark.stop()
