#!/bin/bash
source ./env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /p2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /p2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal $1 /p2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./problem2.py hdfs://$SPARK_MASTER:9000/p2/input/
