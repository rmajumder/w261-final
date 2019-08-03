import numpy as np
import pandas as pd
import time

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
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

sc_conf = SparkConf()
sc_conf.setAppName("w261_final_rishi")

spark = SparkSession\
        .builder\
        .config(conf=sc_conf)\
        .getOrCreate()
    
sc = spark.sparkContext

def print_to_output_file(p_txt):
    print(p_txt)
    
print_to_output_file("Test write---------->")

def fill_missing_val(t_data):
    #Fill missing values - numerical
    return t_data.na.fill(0)

def get_columns():
    num_cols = []
    cat_cols = []
    
    for c in range(2, 14):
        num_cols.append("_" + str(c))
        
    for c in range(14, 41):
        cat_cols.append("_" + str(c))
    
    return num_cols, cat_cols

def get_vect_assem(t_data, cols):    
    assembler = VectorAssembler(inputCols=cols, outputCol="features")

    return assembler.transform(t_data)


def assemble_all_dfs():
    
    num_cols, cat_cols = get_columns()
    cat_indexed_col = []
    
    df_train = spark.read.csv("gs://w261hw5rishi/projdata/split_df_indx/train_p_output.csv", header = True, inferSchema = True)
    df_train = df_train.withColumn("id", monotonically_increasing_id())
    
    for nc in num_cols:
        df1 = spark.read.csv("gs://w261hw5rishi/projdata/split_df/train" + nc + ".csv", header = True, inferSchema = True)
        df1 = df1.withColumn("id", monotonically_increasing_id())
        df_train = df_train.join(df1, "id", "outer")
        
    for c in cat_cols:
        df1 = spark.read.csv("gs://w261hw5rishi/projdata/split_df_indx/train_p_" + c + ".csv", header = True, inferSchema = True)
        df1 = df1.withColumn("id", monotonically_increasing_id())
        df_train = df_train.join(df1, "id", "outer")
        cat_indexed_col.append(df1.columns[0])
    
    df_train = df_train.drop('id')
    
    all_c = num_cols + cat_indexed_col
    #print(all_c)
    
    df_train = fill_missing_val(df_train)
    
    df_train = get_vect_assem(df_train, all_c).select(['features', 'output'])
    
    
    return df_train
    
    
def run_standard_scaler(t_data):
    standardscaler=StandardScaler().setInputCol("features").setOutputCol("scaled_features")
    t_data = standardscaler.fit(t_data).transform(t_data)
    
    return t_data

#Train test split

def train_test_split(t_data):
    train, test = t_data.randomSplit([0.7, 0.3], seed = 2019)
    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))
    
    return train, test

def feature_selection(t_data):
    #Feature selection
    css = ChiSqSelector(featuresCol='scaled_features',outputCol='Aspect',labelCol='output',numTopFeatures=10)
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
    print_to_output_file('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
    
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    
    trainingOutput = "Accuracy: " + str(accuracy) + "\nFPR: " + str(falsePositiveRate) + "\nTPR: " + str(truePositiveRate) + "\nF-measure: " + str(fMeasure) + "\nPrecision: " + str(precision) + "\nRecall: " + str(recall)
    
    print_to_output_file(trainingOutput)
    
def print_perf_eval(predictions):
    predictions.filter(predictions['prediction'] == 0) \
    .select("probability","output","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
    
    evaluator = BinaryClassificationEvaluator(labelCol = 'output')
    print_to_output_file("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    
def run_logistic_regression(tn_data):
    lr = LogisticRegression(maxIter=10, featuresCol="scaled_features", labelCol="output", 
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
    print_to_output_file("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    
    

t0 = time.time()

df_tn = assemble_all_dfs()
t1 = time.time()
print_to_output_file('Runtime - assemble all' + str(float(t1 - t0)))


t0 = time.time()

df_tn = run_standard_scaler(df_tn)

t1 = time.time()
print_to_output_file('Runtime - standard scaler' + str(float(t1 - t0)))


#t0 = time.time()

#df_tn = feature_selection(df_tn)

#t1 = time.time()
#print_to_output_file('Runtime - feature selection' + str(float(t1 - t0)))

t0 = time.time()

train, test = train_test_split(df_tn)

t1 = time.time()
print_to_output_file('Runtime - test train split' + str(float(t1 - t0)))


t0 = time.time()

train = get_balance_weight_ratio_data(train)

t1 = time.time()
print_to_output_file('Runtime - get balanced train data' + str(float(t1 - t0)))


t0 = time.time()

run_logistic_regression(train)

t1 = time.time()
print_to_output_file('Runtime - lr' + str(float(t1 - t0)))


t0 = time.time()

run_random_forest_algorithm(train, test)

t1 = time.time()
print_to_output_file('Runtime - rf' + str(float(t1 - t0)))


t0 = time.time()

run_gradient_boost(train, test)

t1 = time.time()
print_to_output_file('Runtime - gb' + str(float(t1 - t0)))


t0 = time.time()

run_lsvc(train, test)

t1 = time.time()
print_to_output_file('Runtime - lsvc' + str(float(t1 - t0)))