## Tutorial 101: Decision Tree 
### Understanding the Algorithm &amp; Simple Implementation Code  

The Decision-Tree [decisiontreee.py](/decisiontree.py?raw=true "Decision Tree") is a good example to learn how one of the key mahcine learning algorithms work. For each of the input [**data1** & **data2**](/inputdata.py?raw=true "Input Data") the **createTree** algorithm  generates a decision tree that can be used for prediction purposes. The code presented is based on a modified version of the [original code](https://github.com/pbharrin/machinelearninginaction3x/blob/master/Ch03/trees.py) given by Peter Harrington in his book: **Machine Learning in Action**.

Here shall take a deep dive into how the algorithm works and try to understand how it actually runs. The input data and the graph of the output [decision trees](/output.trees?raw=true "Decision Tree") are depicted in the diagram below. 

<img src="/images/img-1.jpg" width="648" height="338">


## Contents:
- How the Algorithm Works
- Running the Decision-Tree Program in Python
- Part 1: Reading the Data File
- Part 2: Calculating Entropy
- Part 3: Creating Tree - Choosing Tree Root
- Part 3: Looping and Splitting into Subtrees

## How the Algorithm Works
The Decision-Tree algorithm outputs a Python dictionary that represents a tree graph. The dataset contains 3 columns:
a            f         f  
==========   ========  ========
ad            fs        df

two variables: non-surfacing which represents animals that do not need to surfaceIf we run the program with the input data (file) we get the following output: <br>

    # 
    {'non-surfacing': {0: {'flippers': {0: 'maybe', 1: 'no'}}, 1: {'flippers': {0: 'no', 1: 'yes'}}}} 
     
     non-surfacing: 
      | 0: 
      | | flippers: 
      | | | 0: maybe
      | | | 1: no
      | | 
      | 
      | 1: 
      | | flippers: 
      | | | 0: no
      | | | 1: yes
      | | 
      |    
 
Looking at the dict generated from data2 we clearly see we have no-surface see we have _1 has the highest page rank followed by URL_4 and then URL_3 & last URL_2. The algorithm works in the following manner:

- If a URL (page) is referenced the most by other URLs then its rank increases, because being referenced means that it is important which is the case of URL_1. 
- If an important URL like URL_1 references other URLs like URL_4 this will increase the destination’s ranking

Given the above it becomes obvious why URL_4's ranking is higher than the other two URL_2 & URL_3. If we look at the various arrows in the above diagram we can also see that URL_2 is referenced the least and that is why it gets the lowest ranking.

The rest of the article will take a deeper look at the Scala code that implements the algorithm in Spark 2.0. The code looks deceivingly simple but to understand how things actually work requires a deeper understanding of Spark RDDs, Spark's Scala based functional API, as well as Page Ranking formula. The code is made of 3 main parts as shown in the diagram below. The 1st part reads the data file then each URL is given a seed value in rank0. The third part of the code contains the main loop which calculates the contributions by joining the links and ranks data at each iteration and then recalculates the ranks based on that contribution. 

<img src="/images/img-2.jpg" width="806" height="594">

## Running the PageRank Program in Spark
To run the PageRank program you need to pass the class name, jar location, input data file and number of iterations. The command looks like the following (please refer to the [Project Setup Article](https://github.com/abbas-taher/scala-eclipse-spark-hortonwork-project-setup): 

      $ cd /usr/hdp/current/spark2-client
      $ export SPARK_MAJOR_VERSION=2
      $ ./bin/spark-submit --class com.scalaproj.SparkPageRank --master yarn --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1 ~/testing/jars/page-rank.jar /input/urldata.txt 20  
 
In case you would like to use the bash command that comes with Spark you can run the example with the following:

      $ ./bin/run-example SparkPageRank /input/urldata.txt 20

## Part 1: Reading the Data File
The code for the 1st part of the program is as follows:

     (1)    val iters = if (args.length > 1) args(1).toInt else 10   // sets iteration from argument (in our case iter=20)
     (2)    val lines = spark.read.textFile(args(0)).rdd   // read text file into Dataset[String] -> RDD1
            val pairs = lines.map{ s =>
     (3)         val parts = s.split("\\s+")               // Splits a line into an array of 2 elements according space(s)
     (4)              (parts(0), parts(1))                 // create the parts<url, url> for each line in the file
                  }
     (5)    val links = pairs.distinct().groupByKey().cache()   // RDD1 <string, string> -> RDD2<string, iterable>   

The 2nd line of the code reads the input data file and produce a Dataset of strings which are then transformed into an RDD with each line in the file being one entire string within the RDD. You can think of an RDD as a list that is special to Spark because the data within the RDD is distributed among the various nodes. Note that I have introduced a "pairs" variable into the original code to make the program more readable.

In the 3rd line of the code, the split command generates for each line (one entire string) an array with two elements. In the 4th line each of the two elements of the array are accessed and then used to produce a key/value pair. The last line in the code applies the groupByKey command on the key/value pair RDD to produce the links RDD, which is also a key/value pair. Thus, the resultant links RDD for the input data file will be as follows:<br>

&nbsp; Key   &emsp;    Array (Iter)
<br> &nbsp; url_4  &emsp;   [url_3, url_1]
<br> &nbsp; url_3  &emsp;   [url_2, url_1]
<br> &nbsp; url_2   &emsp;  [url_1]
<br> &nbsp; url_1   &emsp;  [url_4]
 
Note that the Array in the above is not a true array it is actually an iterator on the resultant array of urls. This is what the groupByKey command produces when applied on an RDD. This is an important and powerful construct in Spark and every programmer needs to understand it well so that they can use it correctly in their code..

## Part 2: Populating the Ranks Data - Initial Seeds 
 
The code in this part is made of a single line

    var ranks = links.mapValues(v => 1.0)    // create the ranks <key,one> RDD from the links <key, Iter> RDD

The above code creates "ranks0" - a key/value pair RDD by taking the key (URL) from the links RDD and assigning the value = 1.0 to it.  Ranks0 is the initial ranks RDD and it is populated with the seed number 1.0 (please see diagram below). In the 3rd part of the program we shall see how this ranks RDD is recalculated at each iteration and eventually converges, after 20 iterations, into the PageRank probability scores mentioned previously.  

## Part 3: Looping and Calculating Contributions & Recalculating Ranks
 
This part is the heart of the PageRank algorithm. In each iteration, the contributions are calculated and the ranks are recalculated based on those contributions. The algorithm has 4 steps:
<br> &nbsp; 1- Start the algorithm with each page at rank 1
<br> &nbsp; 2- Calculate URL contribution: contrib = rank/size
<br> &nbsp; 3- Set each URL new rank = 0.15 + 0.85 x contrib
<br> &nbsp; 4- Iterate to step 2 with the new rank 

Here is the Spark code for the 4 steps above:

    for (i <- 1 to iters) {
    (1)   val contribs = links.join(ranks)         // join  -> RDD1
    (2)          .values                           // extract values from RDD1 -> RDD2          
    (3)          .flatMap{ case (urls, rank) =>    // RDD2 -> conbrib RDD
                       val size = urls.size        
    (4)                   urls.map(url => (url, rank / size))   // the ranks are distributed equally amongs the various URLs
                 }
    (5)   ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _) // ranks RDD
    }
    
In line 1, the links RDD and the ranks RDD are joined together to form RDD1. Then the values of RDD1 are extracted to form RDD2. In line 3, RDD2 is flatmapped to generate the contrib RDD. Line 4, is a bit tricky to understand. Basically, each URL assigned rank is distributed evenly amongst the URLs it references. The diagram below depicts the various RDD generated and the corresponding key/value pairs produced in the first iteration. 

<img src="/images/img-3.jpg" width="722" height="639">

In the diagram below we depict the contributions and ranks in the first two iterations. In the first iteration, for example URL_3 references URL_1 & URL_2 so it contribution is 1/2 = 0.5 for each of the URLs it references. When the rank is calculated URL_3 get a rank of 0.57 (0.15 + 0.85 * 0.5). The 0.57 rank is then passed to the next contribution cycle. In the second iteration, the contribution of URL_3 is once again split in half 0.57 /2 = 0.285.

<img src="/images/img-4.jpg" width="746" height="477">

At the end of the 20 iterations the resultant ranks converges to the output distribution:

     url_4 has rank: 1.3705281840649928.
     url_2 has rank: 0.4613200524321036.
     url_3 has rank: 0.7323900229505396.
     url_1 has rank: 1.4357617405523626.
 
 ### Concluding Remarks
We can clearly see now after this deep dive that the PageRank sample program that comes with Spark 2.0 looks deceivingly simple. The code is both compact and efficient. To understand how things actually work requires a deeper understanding of Spark RDDs, Spark's Scala based functional API, as well as Page Ranking formula. Programming in Spark 2.0 requires unraveling those RDDs that are implicitly generated on your behalf. 
