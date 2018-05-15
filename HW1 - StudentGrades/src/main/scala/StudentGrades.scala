import org.apache.spark.sql.SparkSession
import java.io._

object StudentGrades {
  def main(args: Array[String]) {
    val whereami = System.getProperty("user.dir")
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val grades1 = whereami + "/Exam_1.txt"
    val grades2 = whereami + "/Exam_2.txt"
    val gradeData1 = spark.read.textFile(grades1).cache().collect()
    val gradeData2 = spark.read.textFile(grades2).cache().collect()
    var gradeExam1 = scala.collection.mutable.ArrayBuffer.empty[Double]
    var gradeExam2 = scala.collection.mutable.ArrayBuffer.empty[Double]
    var finalGradeAverage = scala.collection.mutable.ArrayBuffer.empty[Double]
    var names = scala.collection.mutable.ArrayBuffer.empty[String]
    val printer = new PrintWriter(new File("gradesExport.txt"))
    for (i <- 0 to gradeData1.length-1){
      gradeExam1 += gradeData1(i).toString().split("\\W+")(1).toDouble
      names += gradeData1(i).toString.split("\\W+")(0)
      gradeExam2 += gradeData2(i).toString().split("\\W+")(1).toDouble
    }

    for (i <- 0 to gradeData1.length-1){
      finalGradeAverage += (gradeExam1(i) + gradeExam2(i)) / 2
    }

    for (i <- 0 to names.length-1){
      printer.write(names(i) + ", " + finalGradeAverage(i).toString + "\n")
    }
    printer.close
    println("DONE.")
    spark.stop()
  }
}