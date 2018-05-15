import breeze.linalg._

import scala.collection.mutable.ArrayBuffer

object KMeansProgram {

  def main(args: Array[String]): Unit ={

    def readFile(filename: String): Seq[String] = {
      val bufferedSource = io.Source.fromFile(filename)
      val lines = (for (line <- bufferedSource.getLines()) yield line).toList
      bufferedSource.close
      lines
    }

    def readCSV(): DenseMatrix[Double] ={
      var nums = ArrayBuffer[Double]()
      var csvData = readFile("data.csv")
      var rows = 0
      var cols = 0
      for(line <- csvData){
        var n = line.split(",")
        for(ni <- 0 until n.length){
          nums += n(ni).toDouble
        }
        cols = n.length
        rows += 1
      }
      new DenseMatrix(rows, cols, nums.toArray)
    }

    var dataPoints = readCSV()
    var kmeans = new KMeans
    var centroids = kmeans.createCentroids()
    var totalIterations = 100
    var clusterLabel = kmeans.iterateKMeans(dataPoints, centroids, totalIterations)
    kmeans.printLebalData(clusterLabel)
  }
}
