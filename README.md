## Tutorial 101: Decision Tree 
### Understanding the Algorithm: Simple Implementation Code  

The Python code for a Decision-Tree ([*decisiontreee.py*](/decisiontree.py?raw=true "Decision Tree")) is a good example to learn how one of the key machine learning algorithms work. The [**inputdata.py**](/inputdata.py?raw=true "Input Data") is used by the **createTree** algorithm to generate a simple decision tree that can be used for prediction purposes. The data and code presented here are partially based on an [original version](https://github.com/pbharrin/machinelearninginaction3x/blob/master/Ch03/trees.py) given by Peter Harrington in Chapter 3 of his book: **Machine Learning in Action**.

In this discussion we shall take a deep dive into how the algorithm runs and try to understand its inner workings. The [output graph structure](/output.tree?raw=true "Decision Tree") is depicted in the diagram below. 

<img src="/images/decision-tree.png" width="788" height="472">


## Contents:
- Running the Decision-Tree Program
- Input Dataset Description
- Program Output: The Decision Tree dict
- Traversing Decision Tree: Case Example
- Creating Decsion Tree: How machine learning algorithm works
- Part 1: Calculating Entropy
- Part 2: Calculating Entropy
- Part 3: Creating Tree - Choosing Tree Root
- Part 3: Looping and Splitting into Subtrees

## Running the Decision-Tree Program
To execute the main program you can just run the decisiontree.py file using call to Python via command line:

      $ python decisiontree.py
      
Or you can execute it from within the Python console using the following commands: 

      $ python 
      >>> import decisiontree
      >>> import inputdata
      >>> dataset, features = inputdata.createDataset()
      >>> tree = decisiontree.createTree(dataset, features)
      >>> decisiontree.printTree(tree)
      
## Input Dataset Description
The dataset contains records for 7 species, 2 of which are fish, 3 are not and 2 maybe. The data contains two input feature columns: *non-surfacing* and *flippers* and a 3rd prediction label column: *isfish*. (Note non-surfacing means the specie can survive without coming to surface of the water) 

     dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], 
                [1, 0, 'no'],  [0, 1, 'no'],  [0, 1, 'no'], 
                [1, 1, 'maybe'], [0, 0, 'maybe']]
     
     non-surfacing     flippers      isfish  
    ===============   ==========    ========
       True(1)         True(1)        yes
       True(1)         True(1)        yes
       
       True(1)         False(0)       no
       False(0)        True(1)        no
       False(0)        True(1)        no
       
       True(1)         True(1)        maybe
       False(1)        False(0)       maybe

## Program Output: The Decision Tree dict
The machine learning algorithm outputs a Python dictionary that represents a graph tree. If we run the program with the input data (file) we get the following output (which is idential to the above tree diagram) : <br>

    # output as dict
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

## Traversing Decision Tree: Case example
Here we shall explain how the decision tree relates to the input data and in the following section we shall describe how the tree is created by the machine learning algorithm. 

Looking at the tree (see above diagram) we clearly see that the root node first tests whether the specie is non-surfacing. Then it tests in each case (True or False) if the specie has flippers. There are 4 possible decision cases: maybe, no, no, yes each can be reached based on the given input data. For example:

    A specie isFish = Yes if and only if:
      - non-surfacing = 1
      - flippers = 1

This test runs along the right most branch of the tree and terminates at the yes node at the bottom. Overall using a decision-tree is simple, you take any data record and start traversing the tree based on the values of the feature columns.  For example, the two features of the 7th data records: 
 
    [0, 0, 'maybe']   # 7th data records
    non-surfacing = 0 ; flippers =0 
    isFish = maybe
    
is traversed using the left most branch of the tree because both of its columns are False (0). In this case the terminating node is maybe
   

## Creating Decsion Tree: How machine learning algorithm works
following manner:
 - Each input data is first spit into its feature columns and since the root node in the tree tests the non-surfacing

- If a URL (page) is referenced the most by other URLs then its rank increases, because being referenced means that it is important which is the case of URL_1. 
- If an important URL like URL_1 references other URLs like URL_4 this will increase the destination’s ranking

Given the above it becomes obvious why URL_4's ranking is higher than the other two URL_2 & URL_3. If we look at the various arrows in the above diagram we can also see that URL_2 is referenced the least and that is why it gets the lowest ranking.

The rest of the article will take a deeper look at the Python code that implements the algorithm. The code looks deceivingly simple but to understand how things actually work requires a deeper understanding of recursion, Python's list spliting, as well as Entropy formula. The code is made of 3 main parts as shown in the diagram below. The 1st part reads the data file then each URL is given a seed value in rank0. The third part of the code contains the main loop which calculates the contributions by joining the links and ranks data at each iteration and then recalculates the ranks based on that contribution. 

## Part 1: Calculating Entropy
The code for the calculating entropy in the program is as follows:

       def calculateEntropy(dataSet):
           counter= defaultdict(int)   # number of unique labels and their frequency
     (1)   for vector in dataSet:      
               label = vector[-1]      # always assuming last column is the label column 
               counter[label] += 1
           entropy = 0.0
     (2)   for key in counter:
               probability = counter[key]/len(dataSet)     # len(dataSet) = total number of entries 
               entropy -= probability * log(prob,2)        # log base 2
           return entropy 

The 2nd line of the code reads the input data file and produce a Dataset of strings which are then transformed into an RDD with each line in the file being one entire string within the RDD. You can think of an RDD as a list that is special to Spark because the data within the RDD is distributed among the various nodes. Note that I have introduced a "pairs" variable into the original code to make the program more readable.

In the 3rd line of the code, the split command generates for each line (one entire string) an array with two elements. In the 4th line each of the two elements of the array are accessed and then used to produce a key/value pair. The last line in the code applies the groupByKey command on the key/value pair RDD to produce the links RDD, which is also a key/value pair. 

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
