import os
import subprocess
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from time import time
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.master("local[20]").config("spark.local.dir", "/fastdata/acz19yz").appName("COM6012 Assignment2").getOrCreate()

# hyper-parameter choose.
def getBestParam(cvModel):
    params = cvModel.getEstimatorParamMaps()
    avgMetrics = cvModel.avgMetrics
    all_params = list(zip(params, avgMetrics))
    best_param = sorted(all_params, key=lambda x: x[1], reverse=True)[0]
    return list(best_param[0].values())

# read data.
data = spark.read.csv("Data/HIGGS.csv.gz", inferSchema="true").cache()

# -------------------------------------- data pre-processing using pipeline -----------------------------------------
schemaNames = data.schema.names
ncolumns = len(data.columns)

# dense vector.
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features')
# scaler.
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",withStd=True, withMean=True)

# Creating a pipeline: dataframe => dense vector => scaled dense vector.
pipeline = Pipeline(stages=[assembler, scaler])

# transform data using pipeline.
pipeline_data = pipeline.fit(data)
data_vector = pipeline_data.transform(data)

# column rename.
data_vector = data_vector.withColumnRenamed('_c0', 'labels').select('scaled_features','labels')
data_new = data_vector.withColumnRenamed('scaled_features', 'features')

data.unpersist()

# evaluator using AUC and accuracy.
auc = BinaryClassificationEvaluator(labelCol="labels").setMetricName("areaUnderROC")
acc = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

filename = 'Q1_output.txt'

# -------------------------------------- whole dataset ---------------------------------------------------
tr, te = data_new.randomSplit([0.7, 0.3], 50)

tr.cache()
te.cache()

# random forest.
start = time()
rfc = RandomForestClassifier(labelCol="labels", featuresCol="features", maxDepth=15, numTrees=15) # use hyper-parameter from Q1_code_1_A2.py
model = rfc.fit(tr)
end = time()
predictions = model.transform(te)
result_1_rf = auc.evaluate(predictions)
result_2_rf = acc.evaluate(predictions)

features_rf = model.featureImportances
print("***** rf feature importance: ", features_rf)

running_time_rf = end - start
print("***** rf training time: ", running_time_rf)

# gradient boosted tree.
start = time()
gbc = GBTClassifier(labelCol="labels", featuresCol="features", maxIter=10, maxDepth=10)
model = gbc.fit(tr)
end = time()
predictions = model.transform(te)
result_1_gb = auc.evaluate(predictions)
result_2_gb = acc.evaluate(predictions)

features_gb = model.featureImportances
print("***** gb feature importance: ", features_gb)

running_time_gb = end - start
print("***** gb training time: ", running_time_gb)

with open(filename, 'a') as file_object:
    file_object.write("---------------------------------Q1_2---------------------------------------------------------- \n")
    st = "Random forest AUC score: "+str(result_1_rf)+" Accuracy: "+str(result_2_rf)+" training time: "+str(running_time_rf)+". \n"
    st_features = str(features_rf)+"\n"
    st1 = "GBT AUC score: "+str(result_1_gb)+" Accuracy: "+str(result_2_gb)+" training time: "+str(running_time_gb)+". \n"
    st1_features = str(features_gb)+"\n"

    file_object.write(st)
    file_object.write(st_features)
    file_object.write(st1)
    file_object.write(st1_features)

    file_object.write("--------------------------------------------------------------------------------------------------------- \n")

spark.stop()


