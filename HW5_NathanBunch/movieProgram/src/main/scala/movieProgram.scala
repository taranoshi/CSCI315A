import org.apache.spark._
//import spark.implicits._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Column
import java.io._
import scala.io._
import scala.collection.mutable.ListBuffer
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{rand, col}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.classification.NaiveBayes

/*
DONE: step 1: loop through all the positive / negative reviews and label each (1 = Positive, 2 = Negative)
        during the loop, place the text that is read as a string into a DF.
step 2: check which words are common among different labels and text with each other (possibly remove stop words)
        this will satisfy the TF-IDF requirement
step 3: convert text into vectors and perform regression on the values
step 4: compare accuracy using the actual data (data for the above was using the test folder data)
*/

object Machine {
  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
        d.listFiles.filter(_.isFile).toList.take(10)
    } else {
        List[File]()
    }
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Movie Review Manager").getOrCreate()
    println("Reading data...")
    val df = spark.read.format("csv").option("header", "true").load("movie_data.csv")
    //val df_test = spark.read.format("csv").option("header", "true").load("movie_test.csv")
    //=======================split data ===============================
    //================================= Prep the data for extraction ====================================
    val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\s")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("removed")
    val tokenized = regexTokenizer.transform(df)
    tokenized.show(false)
    val removed = remover.transform(tokenized)
    removed.show(false)
    //======================================start HashingTF and IDF======================================================
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(removed)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("finalIdf")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    val pre_train = rescaledData.select("label", "finalIdf")
    val train_idf = pre_train.withColumnRenamed("finalIdf", "features")
    //================================================================
    //start the word2vec
    val word2Vec = new Word2Vec().setInputCol("removed").setOutputCol("features").setVectorSize(200).setMinCount(0)
    val w2vdf = removed
    val model = word2Vec.fit(w2vdf)
    val result = model.transform(w2vdf)
    val train_w2v = result.select("label", "features") //rescaledData.select("label").withColumn("features", result.col("features"))
    //===============================================================
    //begin classification
    train_idf.show()
    train_w2v.show()
    train_idf.collect
    train_w2v.collect
    //val someCastedDF = (df.columns.toBuffer --= exclude).foldLeft(df)((current, c) =>current.withColumn(c, col(c).cast("float")))
    val exclude = Array("features")
    val castedDF_idf = (train_idf.columns.toBuffer --= exclude).foldLeft(train_idf)((current, c) => current.withColumn(c, col(c).cast("int")))
    val castedDF_w2v = (train_w2v.columns.toBuffer --= exclude).foldLeft(train_w2v)((current, c) => current.withColumn(c, col(c).cast("int")))
    castedDF_idf.show
    castedDF_w2v.show

    val Array(trainingData_idf, testData_idf) = train_idf.randomSplit(Array(0.7, 0.3))
    val Array(trainingData_w2v, testData_w2v) = train_w2v.randomSplit(Array(0.7, 0.3))

    /*val Array(trainingData_idf_bytes, testData_idf_bytes) = castedDF_idf.randomSplit(Array(0.7, 0.3))
    val Array(trainingData_w2v_bytes, testData_w2v_bytes) = castedDF_w2v.randomSplit(Array(0.7, 0.3))
    trainingData_idf_bytes.show
    trainingData_w2v_bytes.show*/

    //val Array(trainingData_idf_vs, testData_idf_vs) = train_idf.randomSplit(Array(0.8, 0.2))
    //val Array(trainingData_w2v_vs, testData_w2v_vs) = train_w2v.randomSplit(Array(0.8, 0.2))
    //classifier 1 - Decision tree
    val labelIndexer_idf_dectree = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_idf)
    val labelIndexer_w2v_dectree = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_w2v)
    val featureIndexer_idf_dectree = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_idf)
    val featureIndexer_w2v_dectree = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_w2v)
    val dt_idf_dectree = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    val dt_w2v_dectree = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    val labelConverter_idf_dectree = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_idf_dectree.labels)
    val labelConverter_w2v_dectree = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_w2v_dectree.labels)
    val pipeline_idf_dectree = new Pipeline().setStages(Array(labelIndexer_idf_dectree, featureIndexer_idf_dectree, dt_idf_dectree, labelConverter_idf_dectree))
    val pipeline_w2v_dectree = new Pipeline().setStages(Array(labelIndexer_w2v_dectree, featureIndexer_w2v_dectree, dt_w2v_dectree, labelConverter_w2v_dectree))
    val model_idf_dectree = pipeline_idf_dectree.fit(trainingData_idf)
    val model_w2v_dectree = pipeline_idf_dectree.fit(trainingData_w2v)
    val predictions_idf_dectree = model_idf_dectree.transform(testData_idf)
    val predictions_w2v_dectree = model_w2v_dectree.transform(testData_w2v)
    val evaluator_idf_dectree = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val evaluator_w2v_dectree = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy_idf_dectree = evaluator_idf_dectree.evaluate(predictions_idf_dectree)
    val accuracy_w2v_dectree = evaluator_w2v_dectree.evaluate(predictions_w2v_dectree)

    //classifier 2 - Random forest classifier
    val labelIndexer_idf_rndtree = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_idf)
    val labelIndexer_w2v_rndtree = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_w2v)
    val featureIndexer_idf_rndtree = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_idf)
    val featureIndexer_w2v_rndtree = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_w2v)
    val rf_idf_rndtree = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
    val rf_w2v_rndtree = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
    val labelConverter_idf_rndtree = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_idf_rndtree.labels)
    val labelConverter_w2v_rndtree = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_w2v_rndtree.labels)
    val pipeline_idf_rndtree = new Pipeline().setStages(Array(labelIndexer_idf_rndtree, featureIndexer_idf_rndtree, rf_idf_rndtree, labelConverter_idf_rndtree))
    val pipeline_w2v_rndtree = new Pipeline().setStages(Array(labelIndexer_w2v_rndtree, featureIndexer_w2v_rndtree, rf_w2v_rndtree, labelConverter_w2v_rndtree))
    val model_idf_rndtree = pipeline_idf_rndtree.fit(trainingData_idf)
    val model_w2v_rndtree = pipeline_w2v_rndtree.fit(trainingData_w2v)
    val predictions_idf_rndtree = model_idf_rndtree.transform(testData_idf)
    val predictions_w2v_rndtree = model_w2v_rndtree.transform(testData_w2v)
    val evaluator_idf_rndtree = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val evaluator_w2v_rndtree = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy_idf_rndtree = evaluator_idf_rndtree.evaluate(predictions_idf_rndtree)
    val accuracy_w2v_rndtree = evaluator_w2v_rndtree.evaluate(predictions_w2v_rndtree)

    //with both below, when I attempt to convert the 'label' column to a numerical....it causes an error. I am not sure how to fix it :/

    //classifier 3 - One vs Rest
    /*val classifier_ovr = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)
    val ovr_ovr = new OneVsRest().setClassifier(classifier_ovr)
    val ovrModel_ovr_idf = ovr_ovr.fit(trainingData_idf_bytes)
    val ovrModel_ovr_w2v = ovr_ovr.fit(trainingData_w2v_bytes)

    val predictions_ovr_idf = ovrModel_ovr_idf.transform(testData_idf_bytes)
    val predictions_ovr_w2v = ovrModel_ovr_w2v.transform(testData_idf_bytes)

    val evaluator_ovr = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val accuracy_ovr_idf = evaluator_ovr.evaluate(predictions_ovr_idf)
    val accuracy_ovr_w2v = evaluator_ovr.evaluate(predictions_ovr_w2v)*/

    //classifier 3 - linear
    /*val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val lrModel_idf = lr.fit(trainingData_idf)
    val lrModel_w2v = lr.fit(trainingData_w2v)
    println(s"Coefficients IDF LINEAR: ${lrModel_idf.coefficients} Intercept: ${lrModel_idf.intercept}")
    println(s"Coefficients W2V LINEAR: ${lrModel_w2v.coefficients} Intercept: ${lrModel_w2v.intercept}")*/

    //classifier 3 - Random forest, more trees
    val labelIndexer_idf_rndtree_max = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_idf)
    val labelIndexer_w2v_rndtree_max = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(train_w2v)
    val featureIndexer_idf_rndtree_max = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_idf)
    val featureIndexer_w2v_rndtree_max = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(train_w2v)
    val rf_idf_rndtree_max = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(30)
    val rf_w2v_rndtree_max = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(30)
    val labelConverter_idf_rndtree_max = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_idf_rndtree_max.labels)
    val labelConverter_w2v_rndtree_max = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer_w2v_rndtree_max.labels)
    val pipeline_idf_rndtree_max = new Pipeline().setStages(Array(labelIndexer_idf_rndtree_max, featureIndexer_idf_rndtree_max, rf_idf_rndtree_max, labelConverter_idf_rndtree_max))
    val pipeline_w2v_rndtree_max = new Pipeline().setStages(Array(labelIndexer_w2v_rndtree_max, featureIndexer_w2v_rndtree_max, rf_w2v_rndtree_max, labelConverter_w2v_rndtree_max))
    val model_idf_rndtree_max = pipeline_idf_rndtree_max.fit(trainingData_idf)
    val model_w2v_rndtree_max = pipeline_w2v_rndtree_max.fit(trainingData_w2v)
    val predictions_idf_rndtree_max = model_idf_rndtree_max.transform(testData_idf)
    val predictions_w2v_rndtree_max = model_w2v_rndtree_max.transform(testData_w2v)
    val evaluator_idf_rndtree_max = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val evaluator_w2v_rndtree_max = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy_idf_rndtree_max = evaluator_idf_rndtree_max.evaluate(predictions_idf_rndtree_max)
    val accuracy_w2v_rndtree_max = evaluator_w2v_rndtree_max.evaluate(predictions_w2v_rndtree_max)

    //results===============================================================================================================================
    println(s"Test Errors DECISION TREE: IDF = ${(1.0 - accuracy_idf_dectree)} | W2V = ${(1.0 - accuracy_w2v_dectree)}")
    println(s"Test Error RANDOM DECISION TREE (10 trees): IDF = ${(1.0 - accuracy_idf_rndtree)} | W2V = ${(1.0 - accuracy_w2v_rndtree)}")
    println(s"Test Error RANDOM DECISION TREE BIG SIZE (30 trees): IDF = ${(1.0 - accuracy_idf_rndtree_max)} | W2V = ${(1.0 - accuracy_w2v_rndtree_max)}")
    //println(s"Test Error NAIVE BYTES: IDF = ${(1 - accuracy_ovr_idf)} | W2V = ${(1 - accuracy_ovr_w2v)}")
    //======================================================================================================================================

    
    spark.stop()
  }
}