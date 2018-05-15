import breeze.linalg._
import breeze.stats._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object LinearRegression {
  def main(args: Array[String]): Unit ={

    def predict(alpha: Double, beta: Double, xi: Double): Double ={
      beta * xi + alpha
    }

    def error(alpha: Double, beta: Double, xi: Double, yi: Double): Double ={
      yi - predict(alpha, beta, xi)
    }

    def sumOfSquaredErrors(alpha: Double, beta: Double, x: DenseVector[Double], y: DenseVector[Double]): Double ={
      var xy = x.data.zip(y.data)
      var s = 0.0
      for(z <- xy){
        s += Math.pow(error(alpha, beta, z._1, z._2), 2)
      }
      s
    }

    def leastSquaresFit(x: DenseVector[Double], y: DenseVector[Double]): (Double, Double) ={
      var beta = correlation(x,y) * stddev(y) / stddev(x)
      var alpha = mean(y) - beta * mean(x)
      (alpha, beta)
    }

    def correlation(a: DenseVector[Double], b: DenseVector[Double]): Double ={
      if (a.length != b.length)
        sys.error("you")

      val n = a.length

      val dot = a.dot(b)
      val adot = a.dot(a)
      val bdot = b.dot(b)
      val amean = mean(a)
      val bmean = mean(b)

      // See Wikipedia http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
      (dot - n * amean * bmean ) / ( Math.sqrt(adot - n * amean * amean)  * Math.sqrt(bdot - n * bmean * bmean) )
    }

    def totalSumOfSquares(y: DenseVector[Double]): Double ={
      var s = 0.0
      for(v <- deMean(y).data){
        s += Math.pow(v,2)
      }
      s
    }

    def deMean(x: DenseVector[Double]): DenseVector[Double] ={
      var xBar = mean(x)
      var newX = ArrayBuffer[Double]()
      for(xi <- x.data){
        newX += xi - xBar
      }
      new DenseVector(newX.toArray)
    }

    def rSquared(alpha: Double, beta: Double, x: DenseVector[Double], y: DenseVector[Double]): Double ={
      1.0 - (sumOfSquaredErrors(alpha, beta, x, y) / totalSumOfSquares(y))
    }

    def squaredError(xi: Double, yi: Double, theta: DenseVector[Double]): Double ={
      var alpha = theta.data(0)
      var beta = theta.data(1)
      Math.pow(error(alpha, beta, xi, yi), 2)
    }

    def squaredErrorGradient(xi: Double, yi: Double, theta: DenseVector[Double]): DenseVector[Double] ={
      var alpha = theta.data(0)
      var beta = theta.data(1)
      DenseVector(-2.0 * error(alpha, beta, xi, yi), -2.0 * xi * error(alpha, beta, xi, yi))
    }

    def minimizeStochastic(target: (Double, Double, DenseVector[Double]) => Double, gradient: (Double, Double, DenseVector[Double]) => DenseVector[Double], x: DenseVector[Double], y: DenseVector[Double], theta0: DenseVector[Double], alpha0: Double = 0.01): DenseVector[Double] ={
      var data = x.data.zip(y.data).toList
      var theta = theta0
      var alpha = alpha0
      var minTheta = DenseVector(0.0,0.0)
      var minValue = Double.PositiveInfinity
      var iterationsWithNoImprovement = 0
      var value = 0.0

      while(iterationsWithNoImprovement < 100){
        value = 0.0
        for(xy <- data){
            value += target(xy._1, xy._2, theta)
        }
        //println(value)
        if (value < minValue){
          minTheta = DenseVector(theta.data(0), theta.data(1))
          minValue = value
          iterationsWithNoImprovement = 0
          alpha = alpha0
        } else {
          iterationsWithNoImprovement += 1
          alpha = alpha * 0.9
        }
        for(xy <- scala.util.Random.shuffle(data)){
          var gradient1 = gradient(xy._1, xy._2, theta)
          theta = theta - (alpha *:* gradient1)
        }
      }
      minTheta
    }

    var numFriendsGood = DenseVector(49.0,41.0,40.0,25.0,21.0,21.0,19.0,19.0,18.0,18.0,16.0,15.0,15.0,15.0,15.0,14.0,14.0,13.0,13.0,13.0,13.0,12.0,12.0,11.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
    var dailyMinutesGood = DenseVector(68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84)

    var ab = leastSquaresFit(numFriendsGood, dailyMinutesGood)
    print("Alpha ")
    println(ab._1)
    print("Beta ")
    println(ab._2)

    print("RSquared ")
    println(rSquared(ab._1, ab._2, numFriendsGood, dailyMinutesGood))

    println("gradient descent")
    var rand = new Random()
    var theta = DenseVector(rand.nextDouble(), rand.nextDouble()) //create two random numbers for this vector
    var result = minimizeStochastic(squaredError, squaredErrorGradient, numFriendsGood, dailyMinutesGood, theta, 0.0001)
    println(result)
  }
}