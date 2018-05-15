import breeze.linalg._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object KMeansProgram {

  def main(args: Array[String]): Unit ={

    def readFile(filename: String): Seq[String] = {
      val bufferedSource = io.Source.fromFile(filename)
      val lines = (for (line <- bufferedSource.getLines()) yield line).toList
      bufferedSource.close
      lines
    }

    def readCSV(): List[DenseVector[Double]] = {
      var returnList = ListBuffer[DenseVector[Double]]()
      var nums = ArrayBuffer[Double]()
      var csvData = readFile("data.csv")
      for(line <- csvData){
        var n = line.split(",")
        for(ni <- 0 until n.length){
          nums += n(ni).toDouble
        }
        returnList += new DenseVector(nums.toArray)
        nums = ArrayBuffer[Double]()
      }
      returnList.toList
    }

    var dataPoints = readCSV()
    var kmeans = KMeans.train(3, 100, dataPoints)
    var results = kmeans.run().toList
    println("Results of KMeans: ")
    for(result <- results){
      print("Cluster Number: ")
      print(result._1)
      print(" ")
      print("Data Point: ")
      var t = result._2.data
      println("[ " + t(0) + ", " + t(1) + " ]")
    }
  }
}