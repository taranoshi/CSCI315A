class rbm (val visible: Int, val hidden: Int){
  val seed = 1234
  var weights = Matrix2D(visible, hidden)
  var rgen = (x: Double, y: Double) => 0.1 * Math.sqrt(6 / (x + y))
  var randomGenerator = new MersenneTwister(1234)
  weights.populate(seed, rgen)
  weights.insert((1,1), 0.0)
  println(weights)

  def train(data: Matrix2D, maxEpochs: Int = 1000, learningRate: Double = 0.1): Unit ={
    var numExamples = data.shape(0)
    data.insert(1, 1.0)

    for(epoch <- 1 to maxEpochs) {
      //println("DATA NxM = " + data.shape(0) + ", " + data.shape(1))
      //println(data)
      var posHiddenActivations = data dot weights
      //var posHiddenActivations = weights dot data
      //println("posHiddenActivations")
      //println(posHiddenActivations)
      var posHiddenProbs = logistic(posHiddenActivations)
     // println("posHiddenProbs")
      //println(posHiddenProbs)
      posHiddenProbs.fix(1, 1)
      //println("posHiddenProbs.fix")
      //println(posHiddenProbs)
      var posHiddenStates = posHiddenProbs > Matrix2D.rand(numExamples, hidden + 1)
      //println("posHiddenStates")
      //println(posHiddenStates)
      var posAssociations = data.T dot posHiddenProbs
      //println("posAssociations")
      //println(posAssociations)

      var negVisibleActivations = posHiddenStates dot weights.T
     // println("negVisibleActivations")
     // println(negVisibleActivations)
      var negVisibleProbs = logistic(negVisibleActivations)
     // println("negVisibleProbs")
     // println(negVisibleProbs)
      negVisibleProbs.fix(1,1)
      //println("negVisibleProbs.fix")
     // println(negVisibleProbs)
      var negHiddenActivations = negVisibleProbs dot weights
     // println("negHiddenActivations")
     // println(negHiddenActivations)
      var negHiddenProbs = logistic(negHiddenActivations)
     // println("negHiddenProbs")
     // println(negHiddenProbs)
      var negAssociations = negVisibleProbs.T dot negHiddenProbs
    //  println("negAssociations")
    //  println(negAssociations)
      weights += (((posAssociations - negAssociations) / numExamples) * learningRate)
     // println("weight UPDATE")
     // println(weights)

      var error = ((data - negVisibleProbs)^2).sum
      println("Epoch " + epoch + ": error is " + error)
    }

  }

  def runVisible(data: Array[Double]): Array[Double] ={
    var numExamples = data.length
    var hiddenStates = Array.fill[Double](data.length)(1.0)
    var data2 = 1.0 +: data
    var hiddenActivations = weights dot data2
    var hiddenProbs = logistic(hiddenActivations)
    hiddenStates = checkArrayLessThan(Array(randomGenerator.nextDouble(), randomGenerator.nextDouble(), randomGenerator.nextDouble()), hiddenProbs)
    hiddenStates = checkArrayLessThan(Array(randomGenerator.nextDouble(), randomGenerator.nextDouble(), randomGenerator.nextDouble()), hiddenProbs)
    //remove the first column of hiddenStates
    hiddenStates = hiddenStates.drop(1)
    hiddenStates
  }

  def runHidden(data: Array[Double]): Array[Double] ={
    var numExamples = data.length
    var data2 = 1.0 +: data
    var visibleActivations = weights.T dot data2 //fix this line...suppose to be: data2 dot weights.T
    var visibleProbs = logistic(visibleActivations)
    var visibleStates = checkArrayLessThan(Array(randomGenerator.nextDouble(), randomGenerator.nextDouble(), randomGenerator.nextDouble()), visibleProbs)
    //remove the bias unit
    visibleStates = visibleStates.drop(1)
    visibleStates
  }

  def daydream(numsamples: Int): Unit ={

  }

  private def logistic(matrix: Matrix2D): Matrix2D ={
    (matrix.exp(-matrix) + 1.0) div 1.0
  }

  private def logistic(vect: Array[Double]): Array[Double] ={
    var vect2 = vect
    for(i <- 0 to vect2.length-1){
      vect2(i) = 1.0 / Math.exp(-1*vect2(i))
    }
    vect2
  }

  private def checkArrayLessThan(first: Array[Double], second: Array[Double]): Array[Double] ={ // >
    var returnArray = Array.fill[Double](first.length)(0.0)
    for(i <- 0 to first.length - 1){
      if (first(i) < second(i)){
        returnArray(i) = 1
      } else {
        returnArray(i) = 0
      }
    }
    returnArray
  }

  private def checkArrayGreaterThan(first: Array[Double], second: Array[Double]): Array[Double] ={ // >
    var returnArray = Array[Double](first.length)
    for(i <- 0 to first.length - 1){
      if (first(i) > second(i)){
        returnArray(i) = 1
      } else {
        returnArray(i) = 0
      }
    }
    returnArray
  }
}

object rbm {
  def apply(visible: Int, hidden: Int): rbm ={
    new rbm(visible, hidden)
  }
}

