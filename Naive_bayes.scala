import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object Naive_bayes extends App {

  // SparkSession
  val spark = SparkSession.builder()
    .appName("DiseaseSymptomClassifier")
    .master("local[*]")
    .config("spark.driver.memory", "8g")
    .getOrCreate()

  // Load CSV file
  val filePath = "C:\\Users\\HP\\IdeaProjects\\DiseaseClassification\\src\\Dataset\\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
  var data = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

  // ========================== Data Preparation ==========================
  val labelColumn = data.columns(0)

  // Convert labels to numeric
  val indexer = new StringIndexer()
    .setInputCol(labelColumn)
    .setOutputCol("indexedLabel")
  data = indexer.fit(data).transform(data)

  // Remove problematic feature
  data = data.drop("regurgitation.1")
  data = data.na.drop()

  val featureCols = data.columns.drop(1) // assemble into a single vector

  // Cast each feature column to DoubleType
  featureCols.foreach { colName =>
    data = data.withColumn(colName, data(colName).cast(DoubleType))
  }

  // More Type-Casting
  val numericCols = data.schema.fields
    .filter(field => field.dataType == DoubleType || field.dataType == IntegerType)
    .map(_.name)
    .filter(_ != "indexedLabel") // exclude the label column

  // Update VectorAssembler to use only numeric columns
  val assembler = new VectorAssembler()
    .setInputCols(numericCols)
    .setOutputCol("features")

  data = assembler.transform(data)

  // ============== CLASS TRIMMING ==============
  val classCounts = data.groupBy("indexedLabel")
    .count()
    .orderBy(desc("count"))

  // Select top 10 classes
  val topClasses = classCounts.limit(10).select("indexedLabel")
  val filteredData = data.join(topClasses, Seq("indexedLabel"), "inner")

  val finalData = filteredData.select("indexedLabel", "features")

  // ======================== MODEL TRAINING STARTS HERE ========================
  val Array(trainingData, testData) = finalData.randomSplit(Array(0.8, 0.2), seed = 1234L)

  println("============================================================================")
  println("============================================================================")
  println("MODEL TRAINING IS STARTING ... ")
  println("80% TRAIN && 20% TEST ")
  println("============================================================================")
  println("============================================================================")

  // Naive Bayes classifier
  val naiveBayes = new NaiveBayes()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")
    .setSmoothing(1.0) // Laplace smoothing
    .setModelType("multinomial") // Model type

  val pipeline = new Pipeline().setStages(Array(naiveBayes))

  val model = pipeline.fit(trainingData) // Train

  val predictions = model.transform(testData) // Test

  // Evaluators
  val accuracyEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val precisionEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("weightedPrecision")

  val recallEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("weightedRecall")

  val f1Evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("f1")

  // ROC-AUC is for binary classification, so we calculate it using BinaryClassificationEvaluator
  val binaryEvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  // Calculate metrics
  val accuracy = accuracyEvaluator.evaluate(predictions)
  val precision = precisionEvaluator.evaluate(predictions)
  val recall = recallEvaluator.evaluate(predictions)
  val f1Score = f1Evaluator.evaluate(predictions)
  val rocAuc = binaryEvaluator.evaluate(predictions)

  // Print metrics
  println("=================================================================================")
  println(s"Test set Accuracy = $accuracy")
  println(s"Test set Precision = $precision")
  println(s"Test set Recall = $recall")
  println(s"Test set F1 Score = $f1Score")
  println(s"Test set ROC-AUC = $rocAuc")
  println("=================================================================================")

  // Confusion Matrix
  val confusionMatrix = predictions
    .select("indexedLabel", "prediction")
    .groupBy("indexedLabel")
    .pivot("prediction")
    .count()
    .na.fill(0)

  println(s"Confusion Matrix : ")
  confusionMatrix.show()

}
