import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.io._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ArrayBuffer

object Machine {

    def readFile(filename: String): Seq[String] = {
      val bufferedSource = Source.fromFile(filename)
      val lines = (for (line <- bufferedSource.getLines()) yield line).toList
      bufferedSource.close
      lines
    }

    def readCSV(): Array[org.apache.spark.mllib.linalg.Vector] = {
      var returnList = ArrayBuffer[org.apache.spark.mllib.linalg.Vector]()
      var csvData = readFile("data.csv")
      for(line <- csvData){
        returnList += Vectors.dense(line.split(',').map(_.toDouble))
      }
      returnList.toArray
    }


    def main(args: Array[String]){
        val sc = new SparkContext(new SparkConf().setAppName("KMeans"))
        val parsedData = sc.parallelize(readCSV())

        val numClusters = 3
        val numIterations = 100
        val clusters = KMeans.train(parsedData, numClusters, numIterations)

        val centers = clusters.clusterCenters
        for(c <- centers){
            println(c)
        }

        val WSSSE = clusters.computeCost(parsedData)
        println(s"Within Set Sum of Squared Errors = $WSSSE")
        println(clusters)
    }
}