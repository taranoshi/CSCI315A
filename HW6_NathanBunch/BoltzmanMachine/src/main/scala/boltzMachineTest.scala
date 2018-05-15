object boltzMachineTest extends App {
  var myRbm = rbm(6, 2)
  var testArray = Array.ofDim[Double](6,6)

  testArray = Array(
    Array(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    Array(1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    Array(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    Array(0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
    Array(0.0, 0.0, 1.1, 1.1, 0.0, 0.0),
    Array(0.0, 0.0, 1.1, 1.1, 1.1, 0.0))

  var testData = Matrix2D(testArray)
  myRbm.train(testData, 5000)
  println("Weights:")
  println(myRbm.weights)
  println()
  println("Get the movie category:")
  var userArray = Array.fill[Double](6)(0.0)
  userArray(3) = 1; userArray(4) = 1
  var returnedValues = myRbm.runVisible(userArray)
  println("[" + returnedValues.mkString(" ") + "]")

  /*println("Get the movie preference:")
  var catArray = Array(1.0, 0.0)
  var returned = myRbm.runHidden(catArray)
  println("[" + returned.mkString(" ") + "]")*/
}
