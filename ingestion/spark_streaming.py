from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, to_date
from pyspark.sql.types import StringType
import events_pb2

def decode_proto(events_bytes):
    if events_bytes is None:
        return ""
        
    binary_data = bytes(events_bytes) 
    
    event = events_pb2.UserEvent()
    event.ParseFromString(binary_data)
    
    return f"{event.user_id},{event.item_id},{event.action},{event.timestamp}"

decode_udf = udf(decode_proto, StringType())

def get_spark_session():
    return SparkSession.builder \
        .appName("ML-Platform-Ingestion") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "admin") \
        .config("spark.hadoop.fs.s3a.secret.key", "password123") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .getOrCreate()

spark = get_spark_session()

df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribePattern", "user_.*") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()


processed_df = df.select(
    decode_udf(col("value")).alias("raw"),
    col("topic")
) \
.selectExpr("split(raw, ',') as parts", "topic") \
.select(
    col("parts")[0].alias("user_id"),
    col("parts")[1].alias("item_id"),
    col("parts")[2].alias("action"),
    col("parts")[3].alias("timestamp"),
    col("topic") 
) \
.withColumn("date", to_date(col("timestamp")))

query = processed_df.writeStream \
    .partitionBy("topic", "date") \
    .format("parquet") \
    .option("path", "s3a://training-lake/events/") \
    .option("checkpointLocation", "s3a://training-lake/checkpoints/") \
    .start()

query.awaitTermination()