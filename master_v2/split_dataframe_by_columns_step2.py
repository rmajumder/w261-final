from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

sc_conf = SparkConf()
sc_conf.setAppName("w261_final_rishi")

spark = SparkSession\
        .builder\
        .config(conf=sc_conf)\
        .getOrCreate()
    
sc = spark.sparkContext

df_train = spark.read.csv('gs://w261hw5rishi/projdata/dac/train.csv', header = True, inferSchema = True)

cols = df_train.columns

for col in cols:
    df_with_one_col = df_train.select(col)
    df_with_one_col.write.format("com.databricks.spark.csv").option("header", "true").save("gs://w261hw5rishi/projdata/split_df/train" + col + ".csv")