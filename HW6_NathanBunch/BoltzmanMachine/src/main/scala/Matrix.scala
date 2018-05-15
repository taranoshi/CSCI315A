import scala.util.Random

class Matrix2D (private var sizeX: Int, private var sizeY: Int) {
  private var matrix = Array.ofDim[Double](sizeX, sizeY)
  implicit def bool2int(b:Boolean) = if (b) 1 else 0
  var randomGenerator = new MersenneTwister(1234)

  def insert(axis: Int): Unit ={
    if (axis == 0){
      sizeX += 1
      val t = Array.ofDim[Double](sizeX, sizeY)
      matrix = copy((1,0), matrix, t)
    } else {
      sizeY += 1
      val t = Array.ofDim[Double](sizeX, sizeY)
      matrix = copy((0,1), matrix, t)
    }
  }

  def insert(axis: Int, value: Double): Unit ={
    if (axis == 0){
      sizeX += 1
      val t = Array.fill(sizeX, sizeY)(value)
      matrix = copy((1,0), matrix, t)
    } else {
      sizeY += 1
      val t = Array.fill(sizeX, sizeY)(value)
      matrix = copy((0,1), matrix, t)
    }
  }

  def insert(offset: (Int, Int)): Unit ={
    sizeX = sizeX + offset._1
    sizeY = sizeY + offset._2
    val t = Array.ofDim[Double](sizeX, sizeY)
    matrix = copy(offset, matrix, t)
  }

  def insert(offset: (Int, Int), value: Double): Unit ={
    sizeX = sizeX + offset._1
    sizeY = sizeY + offset._2
    val t = Array.fill(sizeX, sizeY)(value)
    matrix = copy(offset, matrix, t)
  }

  def remove(column: Int): Matrix2D ={ //may cause problems
    var L = matrix.toList
    if (L.size < column) L
    else L = L.take(column) ++ L.drop(column+1)
    var t = L.toArray
    Matrix2D(t)
  }

  private def copy(offset: (Int, Int), source: Array[Array[Double]], destination: Array[Array[Double]]): Array[Array[Double]] ={
    var dx = Math.abs(0 - offset._1)
    var dy = Math.abs(0 - offset._2)
    for(i <- offset._1 to destination.length - 1){
      for (j <- offset._2 to destination(1).length - 1){
        destination(i)(j) = source(i - dx)(j - dy)
      }
    }
    destination
  }

  private def set(a: Array[Array[Double]]) = {matrix = a}

  def populate(seed: Int): Unit ={
    var r = new Random(seed)
    for(i <- 0 to matrix.length - 1){
      for(j <- 0 to matrix(0).length - 1){
        matrix(i)(j) = Random.nextDouble()
      }
    }
  }

  def fill(value: Double): Unit ={
    for(i <- 0 to matrix.length - 1){
      for(j <- 0 to matrix(0).length - 1){
        matrix(i)(j) = value
      }
    }
  }

  def populate(seed: Int, generator: (Double, Double) => Double): Unit ={
    //double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
    var min = -generator(sizeX, sizeY)
    var max = generator(sizeX, sizeY)
    var r = new Random(seed)
    for(i <- 0 to matrix.length - 1){
      for(j <- 0 to matrix(0).length - 1){
        matrix(i)(j) = min + (max - min) * randomGenerator.nextDouble()//Random.nextDouble()
      }
    }
  }

  def populate: Unit ={
    var r = Random
    for(i <- 0 to matrix.length - 1){
      for(j <- 0 to matrix(0).length - 1){
        matrix(i)(j) = Random.nextDouble()
      }
    }
  }

  def dot(other: Matrix2D): Matrix2D ={
    if (this.shape(1) != other.shape(0)){
      throw new IndexOutOfBoundsException("Matrices were not the right shape! [" + this.shape(1) + " != " + other.shape(0) + "]")
    }
    val n = this.shape(1)
    var a = matrix.clone()
    var b = other.matrix.clone()
    var c = Array.ofDim[Double](this.shape(0), other.shape(1))

    for(i <- 0 until c.length){
      for (j <- 0 until c(0).length){
        for (k <- 0 until n){
          c(i)(j) += a(i)(k) * b(k)(j)
        }
      }
    }
    Matrix2D(c)
  }

  def dot(other: Array[Double]): Array[Double] ={
    var array = Array.fill[Double](this.shape(0))(0.0)
    if (this.shape(0) != other.length){
      throw new IndexOutOfBoundsException("Matrices were not the right shape! [" + this.shape(1) + " != " + other.length + "]")
    } else {
      for(i <- 0 to matrix(0).length - 1){
        var tmp = 0.0
        for (j <- 0 to matrix.length - 1){
          tmp += other(j)*matrix(j)(i)
        }
        array(i) = tmp
      }
      array
    }
  }

  def unary_-(): Matrix2D ={
    var t = matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = -1.0 * t(i)(j)
    }
    Matrix2D(t)
  }

  def >(other: Matrix2D): Matrix2D = { //possibly fix this
    var om = other.matrix.clone()
    var newMatrix = Array.ofDim[Double](matrix.length, matrix(0).length)
    for(i <- 0 to matrix.length - 1){
      for (j <- 0 to matrix(0).length - 1){
        try {
          if (matrix(i)(j) > om(i)(j)) {
            newMatrix(i)(j) = 1
          } else {
            newMatrix(i)(j) = 0
          }
        } catch {
          case e: Exception =>{}
        }
      }
    }
    Matrix2D(newMatrix)
    /*var t = matrix
    t.zip(other.matrix) map { case(te, oe) => te.zip(oe).map { case (a,b) => (a > b):Int}}
    Matrix2D(t)*/
  }

  def -(other: Matrix2D): Matrix2D = {
    var t = Array.ofDim[Double](matrix.length, matrix(0).length)
    for(i <- 0 to t.length - 1){
      for(j <- 0 to t(0).length - 1) {
        t(i)(j) = matrix(i)(j) - other.matrix(i)(j)
      }
    }
    Matrix2D(t)
  }

  def /(value: Double): Matrix2D ={
    var t = matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = t(i)(j) / value
    }
    Matrix2D(t)
  }

  def *(value: Double): Matrix2D ={
    var t = matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = t(i)(j) * value
    }
    Matrix2D(t)
  }

  def ^(value: Int): Matrix2D = {
    var t = matrix.clone()
    for(i <- 0 to t.length - 1){
      for(j <- 0 to t(0).length - 1){
        for (k <- 0 to value - 2) {
          t(i)(j) = t(i)(j) * t(i)(j)
        }
      }
    }
    Matrix2D(t)
  }

  def T: Matrix2D ={
    var t = matrix.clone().transpose
    Matrix2D(t)
  }

  def exp(matrix: Matrix2D): Matrix2D ={
    var t = matrix.matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = Math.exp(t(i)(j))
    }
    Matrix2D(t)
  }

  def div(value: Double): Matrix2D ={
    var t = matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = value / t(i)(j)
    }
    Matrix2D(t)
  }

  def +(value: Double): Matrix2D = {
    var t = matrix.clone()
    for(i <- 0 to t.length - 1; j <- 0 to t(0).length - 1){
      t(i)(j) = t(i)(j) + value
    }
    Matrix2D(t)
  }

  def +(other: Matrix2D): Matrix2D = {
    var l = Array.ofDim[Double](matrix.length, matrix(0).length)
    for(i <- 0 to matrix.length - 1){
      for(j <- 0 to matrix(0).length - 1) {
        l(i)(j) = matrix(i)(j) + other.matrix(i)(j)
      }
    }
    Matrix2D(l)
  }

  def shape(axis: Int): Int = {
    if (axis == 0)
      matrix.length
    else
      matrix(0).length
  }

  def fix(axis: Int, value: Double): Unit ={
    if (axis == 0){
      for(i <- 0 to matrix(0).length - 1){
        matrix(0)(i) = value
      }
    }

    if (axis == 1){
      for(j <- 0 to matrix.length - 1){
        matrix(j)(0) = value
      }
    }
  }

  def sum: Double ={
    var s = 0.0
    for(i <- 0 to matrix.length - 1){
      for (j <- 0 to matrix(0).length - 1){
        s += matrix(i)(j)
      }
    }
    s
  }

  override def toString: String = {
    var s = "["
    for(row <- matrix){
      s+="["
      for (cell <- row){
        s += cell
        s += " "
      }
      s += "]\n"
    }
    s+= "]"
    s
  }
}

object Matrix2D  {
  def apply(X: Int, Y: Int): Matrix2D ={
    new Matrix2D(X,Y)
  }

  def apply(array: Array[Array[Double]]): Matrix2D ={
    var t = new Matrix2D(array.length, array(0).length)
    t.set(array)
    t
  }


  def rand(x: Int, y: Int): Matrix2D = {
    var t = new Matrix2D(x,y)
    var rgen = (x: Double, y: Double) => 0.1 * Math.sqrt(6 / (x + y))
    t.populate(1234, rgen)
    t
  }
}
