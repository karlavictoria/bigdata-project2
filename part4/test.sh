#!/bin/bash
../start.sh
source ../env.sh
hdfs dfsadmin -safemode leave
/usr/local/hadoop/bin/hdfs dfs -rm -r /project2/part4/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /project2/part4/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/adultdata.csv /project2/part4/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/adulttest.csv /project2/part4/

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./part4.py hdfs://$SPARK_MASTER:9000/project2/part4/adultdata.csv hdfs://$SPARK_MASTER:9000/project2/part4/adulttest.csv
../stop.sh
