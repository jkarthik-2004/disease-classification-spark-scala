import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions.desc

object Logistic extends App {

  // Initialize SparkSession
  val spark = SparkSession.builder()
    .appName("DiseaseSymptomClassifier")
    .master("local[*]")
    .config("spark.driver.memory", "5g")
    .getOrCreate()

  // Load CSV file into DataFrame
  val filePath = "C:\\Users\\HP\\IdeaProjects\\DiseaseClassification\\src\\Dataset\\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
  var data = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

  // Data Preparation
  val labelColumn = data.columns(0)

  // Convert labels to numeric
  val indexer = new StringIndexer()
    .setInputCol(labelColumn)
    .setOutputCol("indexedLabel")
  data = indexer.fit(data).transform(data)

  // Remove problematic feature and handle missing values
  data = data.drop("regurgitation.1")
  data = data.na.drop()

  val featureCols = data.columns.drop(1) // exclude the label column

  // Cast each feature column to DoubleType
  featureCols.foreach { colName =>
    data = data.withColumn(colName, data(colName).cast(DoubleType))
  }

  // Use VectorAssembler to assemble the feature columns into a single vector
  val assembler = new VectorAssembler()
    .setInputCols(featureCols.drop(1)) // exclude label column
    .setOutputCol("features")

  data = assembler.transform(data)

  // Trim dataset to top 10 most frequent classes
  val classCounts = data.groupBy("indexedLabel").count().orderBy(desc("count"))
  val topClasses = classCounts.limit(10).select("indexedLabel")
  val filteredData = data.join(topClasses, Seq("indexedLabel"), "inner")
  val finalData = filteredData.select("indexedLabel", "features")

  // Split data into training and test sets
  val Array(trainingData, testData) = finalData.randomSplit(Array(0.8, 0.2), seed = 1234L)

  // Logistic Regression model
  val logisticRegression = new LogisticRegression()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  // Set up the pipeline
  val pipeline = new Pipeline().setStages(Array(logisticRegression))

  // Train the model
  val model = pipeline.fit(trainingData)

  // Make predictions
  val predictions = model.transform(testData)

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

  // ROC-AUC is used for binary classification. We can calculate it for each class or use the first class.
  val rocEvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  // Calculate metrics
  val accuracy = accuracyEvaluator.evaluate(predictions)
  val precision = precisionEvaluator.evaluate(predictions)
  val recall = recallEvaluator.evaluate(predictions)
  val f1Score = f1Evaluator.evaluate(predictions)

  // For ROC-AUC, you would generally need a binary classifier, so this metric may not apply directly to multiclass classification.
  // You can still calculate it, but it may not be meaningful in this multiclass setting.
  val rocAuc = rocEvaluator.evaluate(predictions)

  // Print evaluation metrics
  println("=================================================================================")
  println(s"Test set Accuracy = $accuracy")
  println(s"Test set Precision = $precision")
  println(s"Test set Recall = $recall")
  println(s"Test set F1 Score = $f1Score")
  println(s"Test set ROC-AUC = $rocAuc")
  println("=================================================================================")

  // Stop Spark session
  spark.stop()
}
