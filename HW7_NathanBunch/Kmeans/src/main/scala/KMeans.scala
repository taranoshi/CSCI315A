import breeze.linalg._
import scala.collection.mutable._
import scala.util.Random

class KMeans private(private var k:Int, private var maxIterations:Int, private var data: List[DenseVector[Double]]){
  var iterations = 0

  private var centroids = List[DenseVector[Double]]()

  private var cluster = ListBuffer[(Int,DenseVector[Double])]()

  def run():ListBuffer[(Int,DenseVector[Double])] = {

    while(iterations< maxIterations)
    {
      if(centroids.length>0)
      {
        centroids = calculateCentroid(cluster)
        cluster = ListBuffer[(Int,DenseVector[Double])]()
      }
      else
      {
        centroids = initializeCentroids(data)
      }

      data.foreach(d =>
      {
        var b = getTheClosestCentroid(centroids,d)
        cluster += b
      })

      iterations+=1
    }

    cluster

  }

  private def eucledianDistance(x:DenseVector[Double],y:DenseVector[Double]):Double = {
    val sub = x-:-y
    sub.t * sub
  }

  private def getTheClosestCentroid(centroids:List[DenseVector[Double]],row:DenseVector[Double]):(Int,DenseVector[Double]) = {
    var min = -1.0
    var index = 0
    var closestCentroid = -1
    centroids.foreach(centroid => {
      index+=1
      var distance = eucledianDistance(centroid,row)
      if(min>distance||min == -1.0)
      {
        min = distance
        closestCentroid = index
      }
    })
    (closestCentroid,row)
  }

  private def calculateCentroid(cluster:ListBuffer[(Int,DenseVector[Double])]):List[DenseVector[Double]] = {
    var clusterCount = 2
    var meanTemp = ListBuffer[DenseVector[Double]]()
    cluster.groupBy(_._1).foreach((c)=>{
      val sum = c._2.reduce((a,b)=>(a._1,(a._2+b._2)))
      meanTemp += (sum._2 :/ c._2.length.toDouble)
    })
    meanTemp.toList
  }

  private def initializeCentroids(data : List[DenseVector[Double]]):List[DenseVector[Double]] = {
    Random.shuffle(data).take(k)
  }

}

object KMeans {

  def train(k:Int, maxIterations:Int,data: List[DenseVector[Double]]):KMeans = {
    new KMeans(k,maxIterations,data)
  }

}

