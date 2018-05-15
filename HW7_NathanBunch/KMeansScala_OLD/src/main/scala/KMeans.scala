import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable.ArrayBuffer

class KMeans {
  def computeEuclideanDistance(point: DenseVector[Double], centroid: DenseVector[Double]): Double ={
    Math.sqrt(sum((point - centroid)^:^2.0))
  }

  def assignLabelCluster(distance: Array[Double], dataPoint: DenseVector[Double], centroids: DenseMatrix[Double]): DenseVector[Double] ={
    var indexOfMinimum = distance.indexOf(distance.min)
    var newVector = ArrayBuffer[Double]()
    newVector += indexOfMinimum*1.0
    for(i <- 0 until dataPoint.length){
      newVector += dataPoint(i)
    }
    for(j <- 0 until centroids(indexOfMinimum, ::).t.toDenseVector.length){
      newVector += centroids(indexOfMinimum, j)
    }
    DenseVector(newVector.toArray)
  }

  def computeNewCentroids(clusterLabel: Double, centroids: DenseVector[Double]): DenseVector[Double] ={
    (clusterLabel +:+ centroids) /:/ 2.0
  }

  def iterateKMeans(dataPoints: DenseMatrix[Double], centroids: DenseMatrix[Double], totalIterations: Int): Array[DenseMatrix[Double]] ={
    var label = DenseVector[Double](0)
    var clusterLabel = DenseMatrix(0.0)
    var totalPoints = dataPoints.rows
    var k = centroids.rows

    for(iteration <- 0 to totalIterations){
      for(indexPoint <- 0 until totalPoints){

        var distance = ArrayBuffer[Double]()

        for(indexCentroid <- 0 until k){
          distance += computeEuclideanDistance(dataPoints(indexPoint, ::).t.toDenseVector, centroids(indexCentroid, ::).t.toDenseVector)
        }

        label = assignLabelCluster(distance.toArray, dataPoints(indexPoint,::).t.toDenseVector, centroids)
        //centroids(label(0)) = computeNewCentroids(label(1), centroids(::, label(0)))
        changeRow(centroids, computeNewCentroids(label(1).toInt, centroids(label(0).toInt, ::).t.toDenseVector), label(0).toInt)

        if (iteration == (totalIterations - 1)){
          appendMatrix(clusterLabel, label) //does nothing
        }
      }
    }
    //return [cluster_label, centroids]
    return Array(clusterLabel, centroids)
  }

  def printLebalData(args: Array[DenseMatrix[Double]]): Unit = {
    var clusterLabel = args(0)
    var centroids = args(1)
    println("Result of K-Means Clustering")
    for(j <- 0 until clusterLabel.rows) {
      print(clusterLabel(j, ::).t.toDenseVector)
      println()
      for (i <- 0 until centroids.rows) {
        print("     ")
        print(centroids(i, ::).t.toDenseVector)
        println()
      }
    }
  }

  def createCentroids(): DenseMatrix[Double] ={
    return DenseMatrix((5.0,0.0), (45.0, 70.0), (50.0, 90.0))
  }

  def appendVector(denseVector: DenseVector[Double], value: Double): DenseVector[Double] ={
    var newDenseVector = DenseVector[Double](denseVector.length + 1)
    for(i <- 0 until denseVector.length){
      newDenseVector(0) = denseVector(i)
    }
    newDenseVector(newDenseVector.length-1) = value
    newDenseVector
  }

  def appendMatrix(denseMatrix: DenseMatrix[Double], value: DenseVector[Double]): DenseMatrix[Double] ={
    var newDenseMatrix = DenseMatrix(denseMatrix.rows + 1, denseMatrix.cols, 0.0)
    for(r <- 0 until denseMatrix.rows){
      for(c <- 0 until denseMatrix.cols){
        newDenseMatrix(r,c) = denseMatrix(r,c)
      }
    }
    for(c <- 0 until newDenseMatrix.cols){
      newDenseMatrix(newDenseMatrix.rows-1, c) = value(c)
    }
    newDenseMatrix
  }

  def changeRow(denseMatrix: DenseMatrix[Double], denseVector: DenseVector[Double], row: Int): DenseMatrix[Double] = { //fix this
    //var newDenseMatrix = denseMatrix.copy
    var cols = denseMatrix.cols
    var rows = denseMatrix.rows+1
    var newDenseMatrix = ArrayBuffer[Double]()
    var oldData = denseMatrix.data
    var addData = denseVector.data
    for(n <- oldData)
      newDenseMatrix += n
    for(n <- addData)
      newDenseMatrix += n
    /*for(i <- 0 until denseMatrix.cols){
      newDenseMatrix(row, i) = denseVector(i)
    }*/
    //denseMatrix
    new DenseMatrix(rows, cols, newDenseMatrix.toArray)
  }
}
