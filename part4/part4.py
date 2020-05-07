from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, mean, col
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
       if len(sys.argv) != 3:
              print("Usage: Decision Tree and Random Forest <file>", file=sys.stderr)
              sys.exit(-1)

       spark = SparkSession\
              .builder\
              .appName("project2part4")\
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

       #Decision Tree model training and evaluation
       dt = DecisionTreeClassifier(labelCol = "class_index",featuresCol = "features")
       dtModel = dt.fit(traindata)
       dttraindata = dtModel.transform(traindata)
       dttestdata = dtModel.transform(testdata)
       dtevaluator = MulticlassClassificationEvaluator(labelCol="class_index", predictionCol = "prediction", metricName="accuracy")
       dttrainaccuracy = dtevaluator.evaluate(dttraindata)
       dttestaccuracy = dtevaluator.evaluate(dttestdata)

       #Random Forest model training and evaluation
       rf = RandomForestClassifier(labelCol = "class_index", featuresCol = "features", numTrees=10)
       rfModel = rf.fit(traindata)
       rftraindata = rfModel.transform(traindata)
       rftestdata = rfModel.transform(testdata)
       rfevaluator = MulticlassClassificationEvaluator(labelCol="class_index", predictionCol = "prediction", metricName="accuracy")
       rftrainaccuracy = rfevaluator.evaluate(rftraindata)
       rftestaccuracy = rfevaluator.evaluate(rftestdata)

       #printing the evaluations
       print("Decision Tree Train accuracy is:",dttrainaccuracy,"Decision Tree Test accuracy is:",dttestaccuracy, \
       "\nRandom Forest Train accuracy is:",rftrainaccuracy,"Random Forest Test accuracy is:",rftestaccuracy)

       spark.stop()
