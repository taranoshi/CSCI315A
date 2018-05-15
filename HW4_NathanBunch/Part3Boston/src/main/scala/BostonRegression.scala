import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

import org.apache.spark.sql.SparkSession

object Machine {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Machine Learn Application").getOrCreate()
    val data = spark.read.format("libsvm").load("/Users/nathanbunch/Datasets/boston_dataset.libsvm") //if on mac
    //do three regressors of your own choice.

    //linear regression ==============================================
    println("Linear Regression")
    val lr_lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lr_lrModel = lr_lr.fit(data)

    println(s"Coefficients: ${lr_lrModel.coefficients} Intercept: ${lr_lrModel.intercept}")

    val lr_trainingSummary = lr_lrModel.summary
    println(s"numIterations: ${lr_trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${lr_trainingSummary.objectiveHistory.mkString(",")}]")
    lr_trainingSummary.residuals.show()
    println(s"RMSE: ${lr_trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${lr_trainingSummary.r2}")

    //decision tree regression ============================================
    println("Decision Tree Regression")
    val treereg_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    val Array(treereg_trainingData, treereg_testData) = data.randomSplit(Array(0.7, 0.3))

    val treereg_dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val treereg_pipeline = new Pipeline()
      .setStages(Array(treereg_featureIndexer, treereg_dt))

    val treereg_model = treereg_pipeline.fit(treereg_trainingData)

    val treereg_predictions = treereg_model.transform(treereg_testData)

    treereg_predictions.select("prediction", "label", "features").show(5)

    val treereg_evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val treereg_rmse = treereg_evaluator.evaluate(treereg_predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + treereg_rmse)

    val treereg_treeModel = treereg_model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treereg_treeModel.toDebugString)

    //random forest regression ====================================================
    println("Random Forest Regression")
    val randfor_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    val Array(randfor_trainingData, randfor_testData) = data.randomSplit(Array(0.7, 0.3))

    val randfor_rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val randfor_pipeline = new Pipeline()
      .setStages(Array(randfor_featureIndexer, randfor_rf))

    val randfor_model = randfor_pipeline.fit(randfor_trainingData)

    val randfor_predictions = randfor_model.transform(randfor_testData)

    randfor_predictions.select("prediction", "label", "features").show(5)

    val randfor_evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val randfor_rmse = randfor_evaluator.evaluate(randfor_predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + randfor_rmse)

    val randfor_rfModel = randfor_model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println("Learned regression forest model:\n" + randfor_rfModel.toDebugString)

    //stop the program
    spark.stop()
  }
}