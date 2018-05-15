//FOR JSON FILES, THEY MUST BE ON A SINGLE LINE!!!!


val studentDF = spark.read.json("/Users/nathanbunch/Google Drive/Programming and Development/Houghton College Classwork/BigData/HW2 - StudentsCSV_JSON/Students.json")
val examDF = spark.read.json("/Users/nathanbunch/Google Drive/Programming and Development/Houghton College Classwork/BigData/HW2 - StudentsCSV_JSON/Exams.json")

studentDF.show()
examDF.show()

studentDF.select("Major").distinct().show()
studentDF.createOrReplaceTempView("students")
var maleStudentData = spark.sql("SELECT * FROM STUDENTS WHERE Gender = 'Male'")
var femaleStudentData = spark.sql("SELECT * FROM STUDENTS WHERE Gender = 'Female'")
maleStudentData.show()
femaleStudentData.show()
var femaleAgeAverage = spark.sql("SELECT SUM(Age) FROM STUDENTS WHERE Gender = 'Female'") 
var femaleAgeCount = spark.sql("SELECT COUNT(*) FROM STUDENTS WHERE Gender = 'Female'")
var femaleAgeAveCalc = femaleAgeAverage.head().get(0).asInstanceOf[Long].toDouble / femaleAgeCount.head().get(0).asInstanceOf[Long].toDouble
println("Female Student Average Age: " + femaleAgeAveCalc)
var maleAgeAverage = spark.sql("SELECT SUM(Age) FROM STUDENTS WHERE Gender = 'Male'") 
var maleAgeCount = spark.sql("SELECT COUNT(*) FROM STUDENTS WHERE Gender = 'Male'")
var maleAgeAveCalc = maleAgeAverage.head().get(0).asInstanceOf[Long].toDouble / maleAgeCount.head().get(0).asInstanceOf[Long].toDouble
println("Male Student Average Age: " + maleAgeAveCalc)
var studentUnder22 = spark.sql("SELECT * FROM STUDENTS WHERE Age < 22")
studentUnder22.show()
var descendingAge = spark.sql("SELECT * FROM STUDENTS ORDER BY Age DESC")
descendingAge.show()
examDF.createOrReplaceTempView("exams")
var exam1Ave = spark.sql("SELECT SUM(Score) FROM EXAMS WHERE Exam = 'exam1'").head().get(0).asInstanceOf[Long].toDouble / spark.sql("SELECT COUNT(*) FROM EXAMS WHERE Exam = 'exam1'").head().get(0).asInstanceOf[Long].toDouble
var exam2Ave = spark.sql("SELECT SUM(Score) FROM EXAMS WHERE Exam = 'exam2'").head().get(0).asInstanceOf[Long].toDouble / spark.sql("SELECT COUNT(*) FROM EXAMS WHERE Exam = 'exam2'").head().get(0).asInstanceOf[Long].toDouble
var exam3Ave = spark.sql("SELECT SUM(Score) FROM EXAMS WHERE Exam = 'exam3'").head().get(0).asInstanceOf[Long].toDouble / spark.sql("SELECT COUNT(*) FROM EXAMS WHERE Exam = 'exam3'").head().get(0).asInstanceOf[Long].toDouble
println("Averages - Exam 1: " + exam1Ave + " Exam 2: " + exam2Ave + " Exam 3: " + exam3Ave)
var examMax1 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam1' ORDER BY Score DESC").head()
println("Exam 1 Max: " + examMax1)
var examMax2 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam2' ORDER BY Score DESC").head()
println("Exam 2 Max: " + examMax2)
var examMax3 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam3' ORDER BY Score DESC").head()
println("Exam 3 Max: " + examMax3)
//find the minimum
var examMin1 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam1' ORDER BY Score ASC").head()
println("Exam 1 Max: " + examMin1)
var examMin2 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam2' ORDER BY Score ASC").head()
println("Exam 2 Max: " + examMin2)
var examMin3 = spark.sql("SELECT * FROM EXAMS WHERE Exam = 'exam3' ORDER BY Score ASC").head()
println("Exam 3 Max: " + examMin3)
var examsDesc = spark.sql("SELECT * FROM EXAMS ORDER BY Exam, Score DESC")
examsDesc.show()