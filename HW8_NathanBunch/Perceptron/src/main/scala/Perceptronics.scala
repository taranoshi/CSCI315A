import breeze.linalg._
import breeze.numerics._

object Perceptronics{
  def main(args: Array[String]) {
    var X = DenseMatrix(
      (-2.0, 4.0, -1.0),
      (4.0, 1.0, -1.0),
      (1.0, 6.0, -1.0),
      (2.0, 4.0, -1.0),
      (6.0, 2.0, -1.0)
    )
    var y = DenseVector(-1.0, -1.0, 1.0, 1.0, 1.0)

    def perceptronSgd(X: DenseMatrix[Double], Y: DenseVector[Double]): DenseVector[Double] = {
      var w = DenseVector(0.0, 0.0, 0.0)
      var epochs = 20

      for (t <- 1 to epochs) {
        for (i <- 0 until X.rows) {
          var m = (X(i, ::) * w) * Y(i)
          if (m <= 0){
            w = w + X(i, ::).t * Y(i)
          }
        }
      }
      w
    }

    var w = perceptronSgd(X,y)
    println(w)
  }
}
