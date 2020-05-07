from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
    if len(sys.argv) != 3:
            print("Usage: logistic regression <file>", file=sys.stderr)
            sys.exit(-1)

    spark = SparkSession\
            .builder\
            .appName("project2part1")\
            .getOrCreate()

    #Data loading and preprocessing 
    # Load the train-test sets
    train = spark.read.csv(sys.argv[1],header =True)

    test = spark.read.csv(sys.argv[2],header =True)

    sqlcontext = SQLContext(spark)
    sqlcontext.registerDataFrameAsTable(train,"train")
    sqlcontext.registerDataFrameAsTable(test,"test")
    train = sqlcontext.sql("""SELECT *
                            FROM train a
                    WHERE a.toxic IN ('0','1')
                        AND a.severe_toxic IN ('0','1')
                        AND a.obscene IN ('0','1')
                        AND a.threat IN ('0','1')
                        AND a.insult IN ('0','1')
                        AND a.identity_hate IN ('0','1')""")

    train = train.na.drop()
    test = test.na.drop()

    for col in train.columns:
        if col in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
            train = train.withColumn(col,train[col].cast(T.FloatType()))

    #Main code
    out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    wordsData = tokenizer.transform(train)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    tf = hashingTF.transform(wordsData)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(tf)
    tfidf = idfModel.transform(tf)
    REG = 0.1
    lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)
    lrModel = lr.fit(tfidf.limit(5000))
    res_train = lrModel.transform(tfidf)
    res_train.select("id", "toxic", "probability", "prediction").show(20)
    res_train.show(5)
    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
    (res_train.withColumn("proba", extract_prob("probability")).select("proba", "prediction") .show())
    test_tokens = tokenizer.transform(test)
    test_tf = hashingTF.transform(test_tokens)
    test_tfidf = idfModel.transform(test_tf)
    test_res = test.select('id')
    test_res.head()
    test_probs = []
    for col in out_cols:
        print(col)
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
        print("...fitting")
        lrModel = lr.fit(tfidf)
        print("...predicting")
        res = lrModel.transform(test_tfidf)
        print("...appending result")
        test_res = test_res.join(res.select('id', 'probability'), on="id")
        print("...extracting probability")
        test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
        test_res.show(5)
    test_res.show(2)

    spark.stop()
