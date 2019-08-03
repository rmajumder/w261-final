from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

sc_conf = SparkConf()
sc_conf.setAppName("w261_final_rishi")

spark = SparkSession\
        .builder\
        .config(conf=sc_conf)\
        .getOrCreate()
    
sc = spark.sparkContext

train_tmp_rdd = sc.textFile('gs://w261hw5rishi/projdata/dac/train.txt')

rdd_train = train_tmp_rdd.map(lambda r : r.split('\t'))

df_train = rdd_train.toDF()

df_train.write.format("com.databricks.spark.csv").option("header", "true").save("gs://w261hw5rishi/projdata/dac/train.csv")