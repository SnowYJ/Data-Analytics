import os
import subprocess
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from time import time
from pyspark.ml.feature import StandardScaler

# maximum threads.
spark = SparkSession.builder.master("local[20]").appName("COM6012 Assignment2").getOrCreate()

# best hyper-parameter choose.
def getBestParam(cvModel):
    params = cvModel.getEstimatorParamMaps()
    avgMetrics = cvModel.avgMetrics
    all_params = list(zip(params, avgMetrics))
    best_param = sorted(all_params, key=lambda x: x[1], reverse=True)[0]
    return list(best_param[0].values())

# read data.
data = spark.read.csv("Data/HIGGS.csv.gz", inferSchema="true")

# -------------------------------------- data pre-processing using pipeline -----------------------------------------
schemaNames = data.schema.names
ncolumns = len(data.columns)

# dense vector.
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns-1], outputCol = 'features')
# scaler.
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",withStd=True, withMean=True)

# Creating a pipeline: dataframe => dense vector => scaled dense vector.
pipeline = Pipeline(stages=[assembler, scaler])

# transform data using pipeline.
pipeline_data = pipeline.fit(data)
data_vector = pipeline_data.transform(data)

# column rename.
data_vector = data_vector.withColumnRenamed('_c0', 'labels').select('scaled_features','labels')
data = data_vector.withColumnRenamed('scaled_features', 'features')

# -------------------------------------- hyper-parameter choosing using cross validator -------------------------------

# split subset into training and test set.
(subset, _) = data.randomSplit([0.05, 0.95], 50)
tr_sub, te_sub = subset.randomSplit([0.7, 0.3], 50)

# random forest & gradient boosting classifier.
rfc = RandomForestClassifier(labelCol="labels", featuresCol="features")
gbc = GBTClassifier(labelCol="labels", featuresCol="features", maxIter=10)

# evaluator using AUC.
auc = BinaryClassificationEvaluator(labelCol='labels').setMetricName("areaUnderROC")

# parameter grid.
paramGrid_rf = ParamGridBuilder().addGrid(rfc.maxDepth, [5, 10, 15]).addGrid(rfc.numTrees, [5, 8, 15]).build()
paramGrid_gb = ParamGridBuilder().addGrid(gbc.maxDepth, [5, 10, 15]).build()

# 3-fold cross validation.
crossval_rf = CrossValidator(estimator=rfc,estimatorParamMaps=paramGrid_rf,evaluator=auc,numFolds=3)
crossval_gb = CrossValidator(estimator=gbc,estimatorParamMaps=paramGrid_gb,evaluator=auc,numFolds=3)

# random forest.
start = time()
cvModel_rf = crossval_rf.fit(tr_sub)
end = time()
predictions = cvModel_rf.transform(te_sub)
result_rf = auc.evaluate(predictions)
print("***** result of rf: ", result_rf)
print("***** time of rf: ", end-start)

running_time_rf = end-start

 best hyper-parameter
best_param_rf = getBestParam(cvModel_rf)
print("***** best hyperparameter: ", best_param_rf)

# training GBT.
start = time()
cvModel_gb = crossval_gb.fit(tr_sub)
end = time()
predictions = cvModel_gb.transform(te_sub)
result_gb = auc.evaluate(predictions)
print("***** result of GBT: ", result_gb)
print("***** time of GBT: ", end-start)

running_time_gb = end-start

best_param_gb = getBestParam(cvModel_gb)
print("***** best hyperparameter: ", best_param_gb)

filename = 'Q1_output.txt'

with open(filename, 'a') as file_object:
    file_object.write("---------------------------------------Q1_1----------------------------------------- \n")
    st_rf = "Random forest running time:"+str(running_time_rf)+" AUC score: "+str(result_rf)+" with maxDepth: "+str(best_param_rf[0])+" numTrees: "+str(best_param_rf[1])+" . \n"
    st_gb = " GBT running time:"+str(running_time_gb)+" AUC score: "+str(result_gb)+" with maxDepth: "+str(best_param_gb[0])+" . \n"
    file_object.write(st_rf)
    file_object.write(st_gb)
    file_object.write("------------------------------------------------------------------------------------ \n")

spark.stop()

