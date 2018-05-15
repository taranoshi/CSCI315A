import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

object Machine {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Machine Learn Application").getOrCreate()
    val data = spark.read.format("libsvm").load("/Users/nathanbunch/Datasets/wine_dataset.libsvm") //if on mac
    //do three classifiers of your own choice.

    //tre classifier =========================================
    println("Tree Classifier ================================")
    val tree_labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    
    val tree_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(3)
      .fit(data)

    val Array(tree_trainingData, tree_testData) = data.randomSplit(Array(0.7, 0.3))
    val tree_dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    
    val tree_labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(tree_labelIndexer.labels)
    
    val tree_pipeline = new Pipeline()
      .setStages(Array(tree_labelIndexer, tree_featureIndexer, tree_dt, tree_labelConverter))
    
    val tree_model = tree_pipeline.fit(tree_trainingData)

    val tree_predictions = tree_model.transform(tree_testData)

    tree_predictions.select("predictedLabel", "label", "features").show(5)

    val tree_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val tree_accuracy = tree_evaluator.evaluate(tree_predictions)
    println("Test Error = " + (1.0 - tree_accuracy))

    val tree_treeModel = tree_model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + tree_treeModel.toDebugString)

    //random forest classifier ==================================================
    println("Random Forest =======================")
    val randfor_labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    val randfor_featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(3)
      .fit(data)

    val Array(randfor_trainingData, randfor_testData) = data.randomSplit(Array(0.7, 0.3))

    val randfor_rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    val randfor_labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(randfor_labelIndexer.labels)

    val randfor_pipeline = new Pipeline()
      .setStages(Array(randfor_labelIndexer, randfor_featureIndexer, randfor_rf, randfor_labelConverter))

    val randfor_model = randfor_pipeline.fit(randfor_trainingData)

    val randfor_predictions = randfor_model.transform(randfor_testData)

    randfor_predictions.select("predictedLabel", "label", "features").show(5)

    val randfor_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val randfor_accuracy = randfor_evaluator.evaluate(randfor_predictions)
    println("Test Error = " + (1.0 - randfor_accuracy))

    val randfor_rfModel = randfor_model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + randfor_rfModel.toDebugString)

    //multilayer perceptron classifier ============================================
    //doesnt work....switch the algprithms




    println("Multiplayer Perceptron =======================")
    val mlp_splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val mlp_train = mlp_splits(0)
    val mlp_test = mlp_splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val mlp_layers = Array[Int](4, 5, 4, 3)

    val mlp_trainer = new MultilayerPerceptronClassifier()
      .setLayers(mlp_layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val mlp_model = mlp_trainer.fit(mlp_train)

    val mlp_result = mlp_model.transform(mlp_test)
    val mlp_predictionAndLabels = mlp_result.select("prediction", "label")
    val mlp_evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + mlp_evaluator.evaluate(mlp_predictionAndLabels))
    //end the program =============================================
    spark.stop()
  }
}