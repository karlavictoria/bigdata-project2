#!/bin/bash
../start.sh
source ../env.sh
hdfs dfsadmin -safemode leave
/usr/local/hadoop/bin/hdfs dfs -rm -r /project2/part1/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /project2/part1/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/train.csv /project2/part1/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/test.csv /project2/part1/

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./part1.py hdfs://$SPARK_MASTER:9000/project2/part1/train.csv hdfs://$SPARK_MASTER:9000/project2/part1/test.csv
../stop.sh
