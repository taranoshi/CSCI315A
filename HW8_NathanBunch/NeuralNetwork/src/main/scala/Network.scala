import breeze.linalg._
import breeze.numerics._
import breeze.numerics.constants._
import breeze.math._

object Network extends App{
    var X = DenseMatrix((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    var y = DenseMatrix((0.0, 1.0, 1.0, 0.0)).t
    var syn0 = 2.0 *:* DenseMatrix.rand[Double](3, 4) -:- 1.0
    var syn1 = 2.0 *:* DenseMatrix.rand[Double](4, 1) -:- 1.0

    for (j <- 1 to 10000) {
      println("Iteration: " + j)
      var l1 = 1.0 /:/ (1.0 +:+ (E ^:^ -(X * syn0)))
      var l2 = 1.0 /:/ (1.0 +:+ (E ^:^ -(l1 * syn1)))
      var l2_delta = (y -:- l2) *:* (l2 *:* (1.0 -:- l2))
      var l1_delta = (l2_delta * syn1.t) *:* (l1 *:* (1.0 -:- l1))
      syn1 += l1.t * l2_delta
      syn0 += X.t * l1_delta
    }

    for (i <- 0 to 3) {
      var l1 = 1.0 /:/ (1.0 +:+ (E ^:^ -(X * syn0)))
      var l2 = 1.0 /:/ (1.0 +:+ (E ^:^ -(l1 * syn1)))
      println("point " + X(i, ::).t + " true label " + y(i, ::).t + " predicted label " + l2(i, ::).t)
    }
}
