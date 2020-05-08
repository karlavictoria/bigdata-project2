from __future__ import print_function
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler
import sys
import numpy as np

features_names = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

if __name__ == "__main__":
        spark = SparkSession.builder.appName("Problem2").getOrCreate()
        
        rawData = spark.read.csv(sys.argv[1], header=True)
        
        # transform all NA string to none type
        cols = [sql.when(~sql.col(x).isin("NA"), sql.col(x)).alias(x)  for x in rawData.columns]
        
        # drop education column and na rows and add constant column
        data = rawData.select(*cols).drop('education').dropna()
        
        # transfer all columns to float type
        data = data.select(*(sql.col(c).cast("float").alias(c) for c in data.columns))
        
        # set ten year CHD to label column
        data = data.withColumnRenamed("TenYearCHD", 'label')
        
        # select features
        maxPValue = 1.0
        while len(features_names) > 0:
                assembler = VectorAssembler(inputCols=features_names, outputCol='features')
                feature_data = assembler.transform(data)
                glr = GeneralizedLinearRegression(family="binomial", link='logit')
                model = glr.fit(feature_data)
                print(model.summary)
                maxPValue = max(model.summary.pValues)
                if maxPValue > 0.05:
                        i = model.summary.pValues.index(maxPValue)
                        print(features_names[i])
                        del features_names[i]
                else:
                        break

        print("final features sets: %s" % features_names)

        # run logistic regression on the selected features with different threshold
        final_data = VectorAssembler(inputCols=features_names, outputCol='features').transform(data)
        train, test = final_data.randomSplit([0.8,0.2])
        
        threshold = [0.5, 0.4, 0.3, 0.2, 0.1]
        for t in threshold:
                lr = LogisticRegression(threshold=t)
                model = lr.fit(train)
                testSummary = model.evaluate(test)
                print("With threshold %s, Accuracy is %s" % (t, testSummary.accuracy))
                print("With threshold %s, Area under roc is %s" % (t, testSummary.areaUnderROC))
                print("With threshold %s, Specificity is %s" % (t, testSummary.recallByLabel[0]))
                print("With threshold %s, Sensitivity is %s" % (t, testSummary.recallByLabel[1]))
        
        spark.stop()
