from pyspark.ml.feature import StringIndexer

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

sc_conf = SparkConf()
sc_conf.setAppName("w261_final_rishi")

spark = SparkSession\
        .builder\
        .config(conf=sc_conf)\
        .getOrCreate()
    
sc = spark.sparkContext

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


def get_all_cat_cols_indexed(t_data, index_col):    
    indexer = StringIndexer(inputCol=t_data.columns[0], outputCol= index_col).setHandleInvalid("keep")
    t_data = indexer.fit(t_data).transform(t_data)
    t_data = t_data.select(index_col)
    return t_data

def pre_process_each_split_df(t_data, index_col):    
    #Fill missing values
    t_data = fill_missing_val(t_data)
    return get_all_cat_cols_indexed(t_data, index_col)


def indexed_all_dfs(cols):
    for c in cols:
        df_train = spark.read.csv("gs://w261hw5rishi/projdata/split_df/train" + c + ".csv", header = True, inferSchema = True)
        index_col = df_train.columns[0] + "Index"
        df_train = pre_process_each_split_df(df_train, index_col)
        df_train.write.format("com.databricks.spark.csv").option("header", "true").save("gs://w261hw5rishi/projdata/split_df_indx/train_p_" + c + ".csv")
        
    
    df_train = spark.read.csv("gs://w261hw5rishi/projdata/split_df/train_1.csv", header = True, inferSchema = True)
    df_train = get_all_cat_cols_indexed(df_train, "output")
    df_train.write.format("com.databricks.spark.csv").option("header", "true").save("gs://w261hw5rishi/projdata/split_df_indx/train_p_output.csv")
        
#Get columns and stages
num_cols, cat_cols = get_columns()

indexed_all_dfs(cat_cols)