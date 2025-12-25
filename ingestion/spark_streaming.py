from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, to_date
from pyspark.sql.types import StringType
import events_pb2

def decode_proto(events_bytes):
    event = events_pb2.UserEvent()
    event.ParseFromString(events_bytes)
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

# Read from Kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "user_events").load()

# Decode and Partition by Date
processed_df = df.select(decode_udf(col("value")).alias("raw")) \
    .selectExpr("split(raw, ',') as parts") \
    .select(
        col("parts")[0].alias("user_id"),
        col("parts")[2].alias("action"),
        col("parts")[3].alias("timestamp")
    ) \
    .withColumn("date", to_date(col("timestamp")))

# Write to S3 (MinIO)
query = processed_df.writeStream \
    .partitionBy("date") \
    .format("parquet") \
    .option("path", "s3a://training-lake/events/") \
    .option("checkpointLocation", "s3a://training-lake/checkpoints/") \
    .start()

query.awaitTermination()