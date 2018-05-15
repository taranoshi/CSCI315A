import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.sql.functions.struct
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier
//import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.source.libsvm._
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

object Machine {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Machine Learn Application").getOrCreate()
    val data = spark.read.format("libsvm").load("/Users/nathanbunch/dataset.libsvm") //if on mac
    val training = spark.read.format("libsvm").load("/Users/nathanbunch/dataset.libsvm") //if on mac
    //val dataset = spark.read.format("libsvm").load("dataset.libsvm") //if on windows
    //val training = spark.read.format("libsvm").load("dataset.libsvm") //if on windows
    data.show()

    //this was a test to see if the data could be manipulated:
    //=====================Binomial logistic regression========================
    val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    // Fit the model
    val lrModel = lr.fit(data)
    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    //=====================Decision Tree Classifier =============================

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    //========================Random Forest Classifier=============================

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val randForest_labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val randForest_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(randForest_trainingData, randForest_testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val randForest_rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val randForest_labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(randForest_labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val randForest_pipeline = new Pipeline()
      .setStages(Array(randForest_labelIndexer, randForest_featureIndexer, randForest_rf, randForest_labelConverter))

    // Train model. This also runs the indexers.
    val randForest_model = randForest_pipeline.fit(randForest_trainingData)

    // Make predictions.
    val randForest_predictions = randForest_model.transform(randForest_testData)

    // Select example rows to display.
    randForest_predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val randForest_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val randForest_accuracy = randForest_evaluator.evaluate(randForest_predictions)
    println("Test Error = " + (1.0 - randForest_accuracy))

    val randForest_rfModel = randForest_model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + randForest_rfModel.toDebugString)

    //============================Random Forest Regression====================
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val ranfg_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(ranfg_trainingData, ranfg_testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val ranfg_rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and forest in a Pipeline.
    val ranfg_pipeline = new Pipeline()
      .setStages(Array(ranfg_featureIndexer, ranfg_rf))

    // Train model. This also runs the indexer.
    val ranfg_model = ranfg_pipeline.fit(ranfg_trainingData)

    // Make predictions.
    val ranfg_predictions = ranfg_model.transform(ranfg_testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val ranfg_evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val ranfg_rmse = ranfg_evaluator.evaluate(ranfg_predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + ranfg_rmse)

    val ranfg_rfModel = ranfg_model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println("Learned regression forest model:\n" + ranfg_rfModel.toDebugString)

    //========================Gradient Boosted Tree Regression========================

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val gradTBr_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(gradTBr_trainingData, gradTBr_testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gradTBr_gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline.
    val gradTBr_pipeline = new Pipeline()
      .setStages(Array(gradTBr_featureIndexer, gradTBr_gbt))

    // Train model. This also runs the indexer.
    val gradTBr_model = gradTBr_pipeline.fit(gradTBr_trainingData)

    // Make predictions.
    val gradTBr_predictions = gradTBr_model.transform(gradTBr_testData)

    // Select example rows to display.
    gradTBr_predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val gradTBr_evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val gradTBr_rmse = gradTBr_evaluator.evaluate(gradTBr_predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + gradTBr_rmse)

    val gradTBr_gbtModel = gradTBr_model.stages(1).asInstanceOf[GBTRegressionModel]
    println("Learned regression GBT model:\n" + gradTBr_gbtModel.toDebugString)

    //=============================================================================
    //==========================COMPARE ALL THE THINGS=============================
    //=============================================================================

    println("Error Percentage for various algorithms:")
    // Print the coefficients and intercept for logistic regression
    println("logistic regression")
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    println()
    //Decision Tree Classifier
    println("Decision Tree Classifier")
    println("Test Error = " + (1.0 - accuracy))
    println()
    //random Forest Classifier
    println("random Forest Classifier")
    println("Test Error = " + (1.0 - randForest_accuracy))
    println()
    //Random forest Regression
    println("Random forest Regression")
    println("Root Mean Squared Error (RMSE) on test data = " + ranfg_rmse)
    println()
    //gradient tree regression
    println("Gradient tree regression")
    println("Root Mean Squared Error (RMSE) on test data = " + gradTBr_rmse)
    println()

    //the only reason that there are only 5 algorithms s becuase whenever I would add more, there would be an
    //aray error, one that had stated it was 'out of space' or 'idex out of bounds or whtever.....'

    spark.stop()
  }
}