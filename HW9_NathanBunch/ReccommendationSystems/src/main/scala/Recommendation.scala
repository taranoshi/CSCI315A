//HW9

import breeze.linalg._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

object Recommendation{
  def main(args: Array[String]) {
    var usersInterest = Array(
      Array("Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"),
      Array("NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"),
      Array("Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"),
      Array("R", "Python", "statistics", "regression", "probability"),
      Array("machine learning", "regression", "decision trees", "libsvm"),
      Array("Python", "R", "Java", "C++", "Haskell", "programming languages"),
      Array("statistics", "probability", "mathematics", "theory"),
      Array("machine learning", "scikit-learn", "Mahout", "neural networks"),
      Array("neural networks", "deep learning", "Big Data", "artificial intelligence"),
      Array("Hadoop", "Java", "MapReduce", "Big Data"),
      Array("statistics", "R", "statsmodels"),
      Array("C++", "deep learning", "artificial intelligence", "probability"),
      Array("pandas", "R", "Python"),
      Array("databases", "HBase", "Postgres", "MySQL", "MongoDB"),
      Array("libsvm", "regression", "support vector machines")
    )

    var popularInterests = usersInterest.flatten.groupBy(identity).mapValues(_.size).toSeq.sortWith(_._2 > _._2).toArray
    //var popularInterests = usersInterest.flatten.groupBy(identity).mapValues(_.size).toSeq.sortBy(_._1).toArray
    //printArray(popularInterests)

    def MostPopularInterests(userInterest: Array[(String, Int)], maxResults: Int = 5): Array[(String, Int)] ={
      var suggestions = ArrayBuffer[(String, Int)]()
      for(interest <- popularInterests){
        for(uInterest <- userInterest){
          if (interest._1 != uInterest._1)
            suggestions += uInterest
        }
      }
      return suggestions.toArray.slice(0, maxResults)
    }

    def CosineSimilarity(v: DenseVector[Double], w: DenseVector[Double]): Double ={
      //where v and w are DenseVector[Double]
      var t = (v.t * w) / math.sqrt((v.t*v) *:* (w.t*w))
      return t
    }
    var UniqueInterests_1 = usersInterest.flatten.distinct.sorted
    var UniqueInterests = new Array[Array[String]](5)
    UniqueInterests(0)=UniqueInterests_1


    def printArray(array: ArrayBuffer[(Int, Double)]): Unit ={
      print("[ ")
      for(elem <- array){
        print(elem + " ")
      }
      println("] ")
    }

    def printArray2(array: Array[String]): Unit ={
      print("[ ")
      for(elem <- array){
        print(elem + " ")
      }
      println(" ]")
    }



    def makeUserInterestVector(user: Array[String]): Array[Double] = {
      //return [1 if interest in userInterest else 0 for interest in uniqueInterest]
      var vectorArray = ArrayBuffer[Array[Double]]()
      var miniArray = ArrayBuffer[Double]()
      for(i <- 0 until UniqueInterests(0).length){
        for(l <- 0 until usersInterest.length){
          if (usersInterest(l) contains UniqueInterests(0)(i)){
            miniArray.append(1.0)
          }
          else miniArray.append(0.0)
        }

      }
      var myDenseDenseVector = miniArray.toArray
      myDenseDenseVector
    }

    var user_interest_matrix_array = usersInterest.map(x => makeUserInterestVector(x))
    var user_interest_matrix = user_interest_matrix_array.map(x=> DenseVector(x))
    var user_similarities_matrix = ArrayBuffer[Double]()
    var user_similarities = ArrayBuffer[Array[Double]]()
    for(interest_vector_j <- user_interest_matrix){
      for(interest_vector_i <- user_interest_matrix){
        user_similarities_matrix += CosineSimilarity(interest_vector_i, interest_vector_j)
      }
      user_similarities += user_similarities_matrix.toArray
      //user_similarities_matrix = ArrayBuffer[Double]()
    }

    def most_similar_users_to(user_id:Int):ArrayBuffer[(Int,Double)]={
      var pairs = ArrayBuffer[(Int,Double)]()
      var other_user_id = 0
      for(similarity <- user_similarities){
        if((user_id != other_user_id) &&(similarity(user_id) > 0)){
          pairs += ((other_user_id, similarity(user_id)))
        }
        other_user_id += 1
      }
      return pairs.sorted
    }
    def user_based_suggestions(user_id:Int, include_current_interest:Boolean=false):Seq[(String,List[Double])] = {
      var suggestions = new HashMap[String,List[Double]].withDefaultValue(Nil)
      for (pairs <- most_similar_users_to(2)) {
        for (interest <- usersInterest(pairs._1)) {
          suggestions(interest) ::= pairs._2
          suggestions(interest) = List(suggestions(interest).sum)
        }
      }
      return suggestions.toSeq.sortBy(-_._2(0))
    }

    var interest_similarities = ArrayBuffer[Array[Double]]()

    var interest_user_matrix = ArrayBuffer[Double]()
    for(interest_vector_j <- user_interest_matrix){
      for(interest_vector_i <- user_interest_matrix){
        interest_user_matrix += CosineSimilarity(interest_vector_i, interest_vector_j)
      }
      interest_similarities += interest_user_matrix.toArray
      interest_user_matrix = ArrayBuffer[Double]()
    }



    def most_similar_interests_to(interest_id:Int):ArrayBuffer[(String,Double)]={
      var pairs = ArrayBuffer[(String,Double)]()
      var other_interest_id = 0
      var similarities = interest_similarities(interest_id)
      for(similarity <- similarities){
        if((interest_id != other_interest_id) &&(similarity > 0)){
          pairs += ((UniqueInterests(0)(other_interest_id), similarity))
        }
        other_interest_id += 1
      }
      return pairs.sorted
    }



    def item_based_suggestions(user_id:Int, include_current_interest:Boolean=false):Seq[(String,List[Double])] = {
      var suggestions = new HashMap[String,List[Double]].withDefaultValue(Nil)
      var user_interest_vector = user_interest_matrix(user_id)
      var interest_id = 0
      for(is_interested <- user_interest_vector){
        if(is_interested == 1){
          var similar_interests = most_similar_interests_to(interest_id)
          for (pairs <- similar_interests) {
            suggestions(pairs._1) ::= pairs._2
            suggestions(pairs._1) = List(suggestions(pairs._1).sum)
          }
          interest_id += 1

        }

      }
      return suggestions.toSeq.sortBy(-_._2(0))
    }

    def printArray3(array: Array[(String, Int)]): Unit ={
      print("[ ")
      for(elem <- array){
        print(elem + " ")
      }
      println(" ]")
    }

      println("Popular Interests")
      printArray3(popularInterests)
      println()

      println("Most Popular New Interests")
      println("already like:, (NoSQL, MongoDB, Cassandra, HBase, Postgres)")
      println()
      //most_popular_new_interests(Array("NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"))
      println()
      println("already like:, (R, Python, statistics, regression, probability)")
      println()
      //  most_popular_new_interests(Array("R", "Python", "statistics", "regression", "probability"))
      println()

      println("User based similarity")
      println("most similar to 0")
      println(most_similar_users_to(0))

      println("Suggestions for 0")
      println()
      println(user_based_suggestions(0))
      println()

      println("Item based similarity")
      println("most similar to 'Big Data'")
      println()
      println(most_similar_interests_to(0))
      println()

      // println("suggestions for user 0")
      //println()
      // println(item_based_suggestions(0))


  }
}