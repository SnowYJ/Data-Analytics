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
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment1") \
    .getOrCreate()

sc = spark.sparkContext


logFile=spark.read.text("Data/access_log_Jul95").cache() # dataframe

# create designed dataframe.
host_pattern = r'(^\S+\.[\S+\.]+\S+)\s' 
ts_date_pattern = r'\[(\d{2}/\w{3}/\d{4})'
ts_time_pattern = r'(:\d{2}:\d{2}:\d{2})'
method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'
status_pattern = r'\s(\d{3})\s'
content_size_pattern = r'\s(\d+)$'

df = logFile.select(regexp_extract('value', host_pattern, 1).alias('host'),
               regexp_extract('value', ts_date_pattern, 1).alias('date'),  
               regexp_extract('value', ts_time_pattern, 1).alias('time_str'),  
               regexp_extract('value', method_uri_protocol_pattern, 1).alias('request_1'),
               regexp_extract('value', method_uri_protocol_pattern, 2).alias('request_2'),
               regexp_extract('value', method_uri_protocol_pattern, 3).alias('request_3'),
               regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
               regexp_extract('value', content_size_pattern, 1).cast('integer').alias('size')
              )

# change time: string to int.
def handle_time(time):
    t = time.split(':')
    if len(t) != 4:
        return -1
    else:
        return int(t[-3]+t[-2]+t[-1])

udf_handle_time = udf(handle_time)

df1 = df.select('*', udf_handle_time(df['time_str']).alias('time') ).drop('time_str')

time_slot_1 = df1.filter(df1['time'].between(0, 35959)).count()/28
time_slot_2 = df1.filter(df1['time'].between(40000, 75959)).count()/28
time_slot_3 = df1.filter(df1['time'].between(80000, 115959)).count()/28
time_slot_4 = df1.filter(df1['time'].between(120000, 155959)).count()/28
time_slot_5 = df1.filter(df1['time'].between(160000, 195959)).count()/27
time_slot_6 = df1.filter(df1['time'].between(200000, 235959)).count()/27

pie_data = [time_slot_1, time_slot_2, time_slot_3, time_slot_4, time_slot_5, time_slot_6]
pie_label = ['0-3:59', '4-7:59', '8-11:59', '12-15:59', '16-19:59', '20-23:59']

# Q1_B.
plt.figure(figsize=(10,10))
explode = [0.01, 0.01, 0.01, 0.2, 0.01, 0.01]
plt.pie(pie_data, labels=pie_label, autopct='%1.1f%%', explode = explode,
radius = 0.8)
plt.title("Q1_B: AVERAGE NUMBER OF REQUEST")
plt.legend(loc='upper right')
plt.savefig('Q1_B')

filename = 'Q1_output.txt'

# Q1_A.
with open(filename, 'a') as file_object:
    file_object.write("---------------------------Q1_A------------------------------ \n")
    for i, p in enumerate(pie_data):
        st = "time: "+pie_label[i]+". the amount of request: "+ str(p) +". \n"
        print(st)
        file_object.write(st)
        file_object.write("------------------------------------------------------------- \n")
        

webpage = df1.filter(df1.request_2.contains('.html'))
webpage_count = webpage.groupby('request_2').count()
webpage_count_order = webpage_count.orderBy(-webpage_count['count'])
top20_webpage = webpage_count_order.head(20)

# Q1_C.
with open(filename, 'a') as file_object:
    file_object.write("---------------------------Q1_C------------------------------ \n")
    for i, p in enumerate(top20_webpage):
        st = str(i+1)+". webpage: "+ p[0] + " count: " + str(p[1]) +". \n"
        print(st)
        file_object.write(st)
        file_object.write("------------------------------------------------------------- \n")

bar_data = [i[1] for i in top20_webpage]
bar_label = [i[0] for i in top20_webpage]
bar_label_new = [i.split('/')[-1] for i in bar_label]

# Q1_D.
plt.figure(figsize=(8, 8))
plt.barh(bar_label_new, bar_data)
plt.title("Q1_D: Top20 Webpage")
plt.savefig('Q1_D')

spark.stop()
