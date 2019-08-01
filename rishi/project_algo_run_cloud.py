import numpy as np
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import when  
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.sql.functions import lit
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# start Spark Session
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

sc_conf = SparkConf()
sc_conf.setAppName("w261_final_rishi")
sc_conf.set('spark.executor.memory', '18g')
sc_conf.set('spark.driver.memory', '18g')
sc_conf.set('spark.executor.cores', '7')
sc_conf.set('spark.driver.cores', '7')

spark = SparkSession\
        .builder\
        .config(conf=sc_conf)\
        .getOrCreate()
    
sc = spark.sparkContext

df_train = spark.read.csv('gs://w261hw5rishi/projdata/dac/train.csv', header = True, inferSchema = True)

def copy_to_output_file(p_txt):
    print(p_txt)
    
copy_to_output_file("Test write---------->")

def fill_missing_val(t_data):
    #Fill missing values - numerical
    return t_data.na.fill(0)


def get_columns(t_data):
    #Store the original columns
    cols = t_data.columns
    
    #Get numeric and categorical column names
    numericCols = t_data.columns[1:14]
    categoricalColumns = t_data.columns[16:41]
    
    return cols, numericCols, categoricalColumns

def match_test_cols_with_train_cols(t_data):
    
    new_names = []
    
    #rename cols to match train data
    for c in t_data.columns:
        curr_pos = c.split('_')[1]
        new_names.append('_' + str(int(curr_pos) + 1))
   
    t_data = t_data.toDF(*new_names)
    
    #include a dummy output col
    t_data = t_data.withColumn("_1", lit(0))
    
    return t_data

def create_string_indexer_vector_assm_pipeline_stages(num_cols, cat_cols):
    stages = []
    indexerCols = []

    for categoricalCol in cat_cols:
        indexerCol = categoricalCol + "Index"
        indexer = StringIndexer(inputCol=categoricalCol, outputCol= indexerCol).setHandleInvalid("keep")
        stages += [indexer]
        indexerCols.append(indexerCol)

    
    label_stringIdx = StringIndexer(inputCol = '_1', outputCol = 'output')
    stages += [label_stringIdx]
    
    assembler = VectorAssembler(inputCols=indexerCols + num_cols, outputCol="features")
    stages += [assembler]
    
    return stages

def run_transformation_pipeline_stages(t_data, cols, stages):
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(t_data)
    t_data = pipelineModel.transform(t_data)

    selectedCols = ['output', 'features'] + cols
    t_data = t_data.select(selectedCols)
        
    return t_data

def run_standard_scaler(t_data):
    standardscaler=StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
    t_data = standardscaler.fit(t_data).transform(t_data)
    
    return t_data

#Train test split

def train_test_split(t_data):
    train, test = t_data.randomSplit([0.7, 0.3], seed = 2019)
    copy_to_output_file("Training Dataset Count: " + str(train.count()))
    copy_to_output_file("Test Dataset Count: " + str(test.count()))
    
    return train, test

def feature_selection(t_data):
    #Feature selection
    css = ChiSqSelector(featuresCol='scaled_features',outputCol='Aspect',labelCol='output',fpr=0.05)
    t_data=css.fit(t_data).transform(t_data)
    
    return t_data

#Imbalance check - train only

def get_balance_weight_ratio_data(t_data):
    dataset_size=float(t_data.select("output").count())
    numPositives=t_data.select("output").where('output == 1').count()
    per_ones=(float(numPositives)/float(dataset_size))*100
    numNegatives=float(dataset_size-numPositives)
    
    #Rebalance data
    BalancingRatio = numNegatives/dataset_size
    
    return t_data.withColumn("classWeights", when(t_data.output == 1,BalancingRatio).otherwise(1-BalancingRatio))

def print_perf_summary(trainingSummary):
    copy_to_output_file('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
    
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    
    trainingOutput = "Accuracy: " + accuracy + "\nFPR: " + falsePositiveRate + "\nTPR: " + truePositiveRate + "\nF-measure: " + fMeasure + "\nPrecision: " + precision + "\nRecall: " + recall
    
    copy_to_output_file(trainingOutput)
    
def print_perf_eval(predictions):
    predictions.filter(predictions['prediction'] == 0) \
    .select("probability","output","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
    
    evaluator = BinaryClassificationEvaluator(labelCol = 'output')
    copy_to_output_file("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    
def run_logistic_regression(tn_data):
    lr = LogisticRegression(maxIter=10, featuresCol="Aspect", labelCol="output", 
                            weightCol="classWeights", predictionCol="prediction")
    # Fit the model
    lrModel = lr.fit(tn_data)

    predict_train=lrModel.transform(tn_data)
    #predict_test=lrModel.transform(ts_data)
    
    trainingSummary = lrModel.summary
    
    print_perf_summary(trainingSummary)
    
def run_random_forest_algorithm(tn_data, ts_data):
    rf = RandomForestClassifier(numTrees=10, featuresCol="scaled_features", labelCol="output", predictionCol="prediction")
    rfModel = rf.fit(tn_data)
    predictions = rfModel.transform(ts_data)
    
    print_perf_eval(predictions)
    
def run_gradient_boost(tn_data, ts_data):
    gbt = GBTClassifier(maxIter=10, featuresCol="scaled_features", labelCol="output", predictionCol="prediction")
    gbtModel = gbt.fit(tn_data)
    predictions = gbtModel.transform(ts_data)
    
    print_perf_eval(predictions)
    
def run_lsvc(tn_data, ts_data):
    sv = LinearSVC(maxIter=10, regParam=0.1,
                     featuresCol="scaled_features", labelCol="output", predictionCol="prediction")
    svModel = sv.fit(tn_data)
    predictions = svModel.transform(ts_data)
    
    evaluator = BinaryClassificationEvaluator(labelCol = 'output')
    copy_to_output_file("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    
#Fill missing values
df_train = fill_missing_val(df_train)

#Get columns and stages
cols, num_cols, cat_cols = get_columns(df_train)
stages = create_string_indexer_vector_assm_pipeline_stages(num_cols, cat_cols)

t0 = time.time()

df_train = run_transformation_pipeline_stages(df_train, cols, stages)

t1 = time.time()
copy_to_output_file('Runtime - transform pipeline: %f seconds' % (float(t1 - t0)))

t0 = time.time()

df_train = run_standard_scaler(df_train)

t1 = time.time()
copy_to_output_file('Runtime - std scaler %f seconds' % (float(t1 - t0)))

t0 = time.time()

df_train = feature_selection(df_train)

t1 = time.time()
copy_to_output_file('Runtime - feature selection %f seconds' % (float(t1 - t0)))

t0 = time.time()

train, test = train_test_split(df_train)

t1 = time.time()
print('Runtime - train_test_split %f seconds' % (float(t1 - t0)))

t0 = time.time()

train = get_balance_weight_ratio_data(train)

t1 = time.time()
print('Runtime - balance weight ratio %f seconds' % (float(t1 - t0)))

t0 = time.time()

run_logistic_regression(train)

t1 = time.time()
print('Runtime - lgr %f seconds' % (float(t1 - t0)))

t0 = time.time()

run_random_forest_algorithm(train, test)

t1 = time.time()
print('Runtime - rf %f seconds' % (float(t1 - t0)))

t0 = time.time()

run_gradient_boost(train, test)

t1 = time.time()
print('Runtime - gb %f seconds' % (float(t1 - t0)))

t0 = time.time()

run_lsvc(train, test)

t1 = time.time()
print('Runtime - lsvc %f seconds' % (float(t1 - t0)))