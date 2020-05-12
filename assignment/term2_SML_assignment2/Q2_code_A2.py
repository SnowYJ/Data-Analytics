import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from time import time
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
import math
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.master("local[20]").config("spark.local.dir", "/fastdata/acz19yz").appName("COM6012 Assignment2").getOrCreate()
    

data = spark.read.csv("Data/train_set.csv.gz", inferSchema="true", header=True).cache()

# --------------------------------------------- dataset pre-processing ---------------------------------------------------
# remove null rows.
df_null = data.replace('?', None)
df = df_null.dropna(how='any')
df_new = df

# convert category value to double type.
def handle_cat(v):
    return ord(v) - 65

udf_handle_cat = udf(handle_cat)

# it seems faster than stringIndexer!!
for i in range(1, 13):
    cat_name = 'Cat' + str(i)
    # column Cat 1 to 12.
    df_new = df_new.withColumn(cat_name, udf_handle_cat(df[cat_name]).cast(DoubleType()))
    
# column NVCat.
df_new = df_new.withColumn('NVCat', udf_handle_cat(df['NVCat']).cast(DoubleType()))

# column OrdCat to double.
df_new = df_new.withColumn('OrdCat', df_new['OrdCat'].cast(DoubleType()))

raw_dataset = df_new.drop('Blind_Model', 'Blind_Submodel', 'Blind_Make')

data.unpersist()
raw_dataset.cache()

# convert to one-hot. Cat 1 to 12 and NVCat
input_columns = ['Cat'+str(i) for i in range(1, 13)] + ['NVCat']
output_columns = ['oh_Cat'+str(i) for i in range(1, 13)] + ['oh_NVCat']

one_hot = OneHotEncoderEstimator(inputCols=input_columns, outputCols=output_columns)
oh_model = one_hot.fit(raw_dataset)
oh_dataset = oh_model.transform(raw_dataset)

raw_dataset.unpersist()
oh_dataset.cache()

# choose feature.
new_schema_name = [i for i in oh_dataset.schema.names if i not in input_columns]
new_schema_name.remove('Row_ID')
new_schema_name.remove('Household_ID')
new_schema_name.remove('Claim_Amount')

# dense vector & scaler.
assembler = VectorAssembler(inputCols = new_schema_name, outputCol = 'features')
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",withStd=True, withMean=True)

# Creating a pipeline: dataframe => dense vector => scaled dense vector.
pipeline = Pipeline(stages=[assembler, scaler])

# transform data using pipeline.
pipeline_data = pipeline.fit(oh_dataset)
data_vector = pipeline_data.transform(oh_dataset)

# column rename.
data_vector = data_vector.withColumnRenamed('Claim_Amount', 'labels').select('scaled_features','labels')
dataset = data_vector.withColumnRenamed('scaled_features', 'features')

oh_dataset.unpersist()

# weight dataset for using logistic regression (only).

data_size = dataset.count()
data_zero_size = dataset.filter(dataset['labels'] == 0).count()
balance_ratio = (data_size - data_zero_size) / data_size

# create weight column.
def balance_weight(v):
    if v != 0:
        return 1-balance_ratio
    else:
        return balance_ratio
    
udf_balance_weight = udf(balance_weight)

# create a weight columns for logistic regression Q2_3.
weight_dataset = dataset.withColumn('weights', udf_balance_weight(dataset['labels']).cast(DoubleType()))

# create label column.
def class_label(v):
    res = 0 if v == 0 else 1
    return res

udf_class_label = udf(class_label)

# create a new column for classfication. 0 or 1
dataset = weight_dataset.withColumn('class_labels', udf_class_label(dataset['labels']).cast(DoubleType()))

dataset.cache()
# ------------------------------------------------ linear regression --------------------------------------------------
(tr, te) = dataset.randomSplit([0.8, 0.2], 50)

tr_non_zero = tr.filter(tr['labels'] != 0)
te_non_zero = te.filter(te['labels'] != 0)

start = time()
lr = LinearRegression(maxIter=50, regParam=0.01, elasticNetParam=0.5, featuresCol='features', labelCol='labels')
lr_model = lr.fit(tr)
end = time()

predictions = lr_model.transform(te)

evaluator_mae = RegressionEvaluator(labelCol="labels", predictionCol="prediction", metricName="mae")
evaluator_mse = RegressionEvaluator(labelCol="labels", predictionCol="prediction", metricName="mse")

mae = evaluator_mae.evaluate(predictions)
mse = evaluator_mse.evaluate(predictions)


filename = 'Q2_output.txt'

with open(filename, 'a') as file_object:
    file_object.write("---------------------------------------Q2_2----------------------------------------- \n")
    st_rf = "regression training time:"+str(end-start)+" mae: "+str(mae)+" mse: "+str(mse)+" . \n"
    file_object.write(st_rf)
    file_object.write("------------------------------------------------------------------------------------ \n")
    

# classifier (logistic regression) + regression (gamma regression).

# training stage.
start = time()
lrc = LogisticRegression(featuresCol='features', labelCol='class_labels', weightCol='weights', maxIter=50, regParam=0.01, elasticNetParam=0.5)
glr = GeneralizedLinearRegression(maxIter=50, regParam=0.01, featuresCol='features', labelCol='labels',  family='gamma', link='log')
lrc_model = lrc.fit(tr)
glr_model = glr.fit(tr_non_zero)
end = time()
print("*** training time: ", end-start)

# testing stage.
prediction = lrc_model.transform(te)
auc = BinaryClassificationEvaluator(labelCol="class_labels").setMetricName("areaUnderROC")
result_auc = auc.evaluate(prediction)
print("*** AUC : ", result_auc)

# remove zero from prediction and labels.
prediction_non_zero = prediction.filter(prediction['prediction'] != 0)
prediction_non_temp_zero = prediction_non_zero.filter(prediction['labels'] != 0)
prediction_non_zero = prediction_non_temp_zero.drop('prediction')

glr_predict_result = glr_model.transform(prediction_non_zero)

mae = RegressionEvaluator(labelCol="labels", predictionCol="prediction", metricName="mae")
mse = RegressionEvaluator(labelCol="labels", predictionCol="prediction", metricName="mse")
mae_res = mae.evaluate(glr_predict_result)
mse_res = mse.evaluate(glr_predict_result)

with open(filename, 'a') as file_object:
    file_object.write("---------------------------------------Q2_3----------------------------------------- \n")
    st_rf = "training time: "+str(end-start)+" AUC score: "+str(result_auc)+" mae: "+str(mae_res)+" mse: "+str(mse_res)+" . \n"
    file_object.write(st_rf)
    file_object.write("------------------------------------------------------------------------------------ \n")

print("*** MAE: ", mae_res)
print("*** MSE: ", mse_res)




