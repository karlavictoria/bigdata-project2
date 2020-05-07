from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, mean, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler


reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
       if len(sys.argv) != 3:
              print("Usage: logistic regression <file>", file=sys.stderr)
              sys.exit(-1)

       spark = SparkSession\
              .builder\
              .appName("project2part3")\
              .getOrCreate()

       #Data loading and preprocessing 
       traindata = spark.read.csv(sys.argv[1],header =True)

       testdata = spark.read.csv(sys.argv[2],header =True)
       #index the categorical values
       indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(traindata) for column in ['workclass','maritalstatus','occupation','relationship','race','sex','nativecountry','class'] ]

       pipeline = Pipeline(stages=indexers)
       traindata = pipeline.fit(traindata).transform(traindata)

       indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") \
       .fit(testdata) for column in ['workclass','maritalstatus','occupation','relationship','race','sex','nativecountry','class'] ]

       pipeline = Pipeline(stages=indexers)

       testdata = pipeline.fit(traindata).transform(testdata)
       #drop the empty values
       traindata = traindata.dropna()
       testdata = testdata.dropna()

       #Now, we drop the columns we don't need

       traindata = traindata.select(traindata.age, traindata.educationnum, \
       traindata.capitalgain, traindata.capitalloss, traindata.hoursperweek, \
       traindata.workclass_index, traindata.maritalstatus_index, \
       traindata.occupation_index, traindata.relationship_index, traindata.race_index, \
       traindata.sex_index, traindata.nativecountry_index, traindata.class_index)

       testdata = testdata.select(testdata.age, testdata.educationnum, \
       testdata.capitalgain, testdata.capitalloss, testdata.hoursperweek, \
       testdata.workclass_index, testdata.maritalstatus_index, \
       testdata.occupation_index, testdata.relationship_index, testdata.race_index, \
       testdata.sex_index, testdata.nativecountry_index, testdata.class_index)

       #We make sure each feature has the correct data type
       for column in traindata.columns:
              traindata = traindata.withColumn(column,traindata[column].cast('float'))

       for column in testdata.columns:
              testdata = testdata.withColumn(column,testdata[column].cast('float'))

       vecAssembler = VectorAssembler(inputCols = traindata.columns[:-1], outputCol = 'features')

       traindata = vecAssembler.transform(traindata)
       testdata = vecAssembler.transform(testdata)

       #Logistic Regression
       lr = LogisticRegression(featuresCol="features", labelCol='class_index')
       lrModel = lr.fit(traindata)
       traindata = lrModel.transform(traindata)
       testdata = lrModel.transform(testdata)

       testdata = testdata.withColumn("testacc",when(testdata.class_index == testdata.prediction, 1).otherwise(0))

       testdata = testdata.withColumn("testacc",testdata["testacc"].cast("float"))
       testacc = testdata.select(mean(col("testacc")).alias("testaccuracy")).collect()
       print("Train accuracy is:",lrModel.summary.accuracy,"\nTest accuracy is:",testacc[0]["testaccuracy"])


       spark.stop()