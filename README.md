## Tutorial 101: Decision Tree 
### Understanding the Algorithm: Simple Implementation Code Example 

The Python code for a Decision-Tree ([*decisiontreee.py*](/decisiontree.py?raw=true "Decision Tree")) is a good example to learn how a basic machine learning algorithm works. The [*inputdata.py*](/inputdata.py?raw=true "Input Data") is used by the **createTree algorithm** to generate a simple decision tree that can be used for prediction purposes. The data and code presented here are a modified version of the [original code](https://github.com/pbharrin/machinelearninginaction3x/blob/master/Ch03/trees.py) given by Peter Harrington in Chapter 3 of his book: **Machine Learning in Action**.

In this discussion we shall take a deep dive into how the algorithm runs and try to understand how the [Python dict tree](/output.tree?raw=true "Decision Tree") structure depicted in the graph below is generated recursively. 

<img src="/images/decision-tree.png" width="709" height="425">


## Contents:
- Historical Note
- Running the Decision-Tree Program
- Input Dataset Description
- Program Output: The Decision Tree dict
- Traversing Decision Tree: Case Example
- Creating Decsion Tree: How machine learning algorithm works
- Part 1: Calculating Entropy
- Part 2: Choosing Best Feature To Branch Tree
- Part 3: Creating Tree - Choosing Tree Root
- Part 3: Looping and Splitting into Subtrees

## Historical Note
Machine learning decision trees were first formalized by [John Ross Quinlan](https://en.wikipedia.org/wiki/Ross_Quinlan) during the years 1982-1985. Along linear and logistic regression, decision trees (with their modern version of random forests) are considered the easiest and the most commonly used machine learning algorithms. 

## Running the Decision-Tree Program
To execute the main function you can just run the decisiontree.py program using a call to Python via the command line:

      $ python decisiontree.py
      
Or you can execute it from within the Python shell using the following commands: 

      $ python 
      >>> import decisiontree
      >>> import inputdata
      >>> dataset, features = inputdata.createDataset()
      >>> tree = decisiontree.createTree(dataset, features)
      >>> decisiontree.pprintTree(tree)
      
## Input Dataset Description
The *createDataset* function generates sample records for 7 species: 2 fish, 3 not fish and 2 maybe. The dataset contains two input feature columns: *non-surfacing* and *flippers* and a 3rd prediction label column: *isfish*. (Note: *non-surfacing* means the specie can survive without coming to the surface of the water) 

     dataset = [[1, 1, 'yes'], [1, 1, 'yes'], 
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
The machine learning program recursively builds a Python dictionary which represents the graph of a tree. If we run the **createTree** function with the input dataset we get the following pretty print output, which is identical to the tree diagram shown above:

    # output as dict
    {'non-surfacing': {0: {'flippers': {0: 'maybe', 1: 'no'}}, 
                       1: {'flippers': {0: 'no', 1: 'yes'}}}} 
     
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
We start first by explaining how the decision tree relates to the input data and in the following sections we shall describe how the tree is created by the machine learning algorithm. 

Looking at the tree diagram we clearly see that the root node first tests whether the specie is non-surfacing. Then it tests in each case (True or False) if the specie has flippers. There are 4 possible decision cases: maybe, no, no, yes each can be reached based on the given input data. For example:

    A specie isFish = Yes if and only if:
      - non-surfacing = 1
      - flippers = 1

This test runs along the right most branch of the tree and terminates at the yes node at the bottom. Overall using a decision-tree is simple, you take any data record and start traversing the tree based on the values of the feature columns.  For example, the two features of the 7th data records: 
 
    [0, 0, 'maybe']   # 7th data records
    non-surfacing = 0 ; flippers = 0 
    isFish = maybe
    
can be used to traverse the left most branch of the tree because both feature columns are False (0). In this case, the branch ends at the *maybe* leaf node. 
   

## Creating Decsion Tree: How machine learning algorithm works
The **createTree algorithm** builds a decision tree recursively. The algorithm is composed of 3 main components:

 - Entropy test to compare information gain in a given data pattern
 - Dataset spliting performed according to the entropy test
 - Growing dict structure to represents the decision tree

In each recursive iteration of the *createTree* function, the algorithm searches for patterns in its given dataset by comparing information gain for each feature. It peforms an entropy test that discriminates between features and then chooses the one that can best split the given dataset into sub-datasets. The algorithm then calls itself recursively to do the pattern search, entropy test and spliting on the new sub-datasets. Recursion terminates and the tree branch is rolled when there are no more features to split in the sub-dataset or when all the prediction labels are the same.

The rest of the article will take a deeper look at the Python code that implements the algorithm. The code looks deceivingly simple but to understand how things actually work requires a deeper understanding of recursion, Python's list spliting, as well as understanding how Entropy works. 

## Part 1: Calculating Entropy
The code for calculating Entropy for the labels in a given dataset:

       def calculateEntropy(dataset):
           counter= defaultdict(int)   # number of unique labels and their frequency
     (1)   for record in dataset:      
               label = record[-1]      # always assuming last column is the label column 
               counter[label] += 1
           entropy = 0.0
     (2)   for key in counter:
               probability = counter[key]/len(dataset)       # len(dataSet) = total number of entries 
               entropy -= probability * log(probability,2)   # log base 2
           return entropy 

There are two main for-loops in the function. Loop (1) calculates the frequency of each label and Loop (2) calculates Entropy for those labels using the below formula:
 
&nbsp; &nbsp; &nbsp; H(X) = - &sum;<sub>i</sub> P<sub>X</sub>(x<sub>i</sub>) * log<sub>2</sub> (P<sub>X</sub>(x<sub>i</sub>))

To compute Entropy H(X) for a given variable X with possible values x<sub>i</sub> we take the negative sum of the product of probability P<sub>X</sub>(x<sub>i</sub>) with the log base 2 of that same probability value. 

Because Entropy uses probability in its formula, it measures **certain disorder** in the data, the greater the Entropy the higher is the disorder in the data. This means, that when a certain dataset includes a high-probability event (data patterns occuring frequently, that event carries less Entropy than when that data pattern occurs less often (low-probability event). For example, if we take the all seven records and measure baseEntropy for all labels we get: 
      
      # labels = [yes,yes,no,no,no,maybe,maybe]
      $ python
      >>> from decisiontree import *
      >>> dataset, features = createDataset()
      >>> baseEntropy = calculateEntropy(dataset)
      >>> print (baseEntropy)
      1.5566567074628228
      
If we drop the last two records and their corresponding *maybe* labels then Entropy decreases because the sample data has lost some variety and thus became more ordered.

      #  labels = [yes,yes,no,no,no]
      >>> entropy = calculateEntropy(dataset[:-2])
      >>> print (entropy)
      0.9709505944546686

If we compute Entropy for a any one record (drop all other 6 records) we get zero Entropy value because there is no variety at all in the data:

      # labels = [yes]
      >>> entropy = calculateEntropy(dataset[:-6])
      >>> print (entropy)
      0.0

## Part 2: Choosing Best Feature to Branch Tree
 
Although, the previous code is used to calculate Entropy for a list of labels, to build the tree top down we need to do more and discriminate between various features and test for their ability to reduce Entropy. 

      def chooseBestFeatureToSplit(dataset):
          baseEntropy = calculateEntropy(dataset)
          bestInfoGain = 0.0; bestFeature = -1

          numFeat = len(dataset[0]) - 1          # do not include last label column     
          for indx in range(numFeat):            # iterate over all the features index
              featValues = {record[indx] for record in dataset}     # put feature values into a set
              featEntropy = 0.0
              for value in featValues:
                  subDataset = splitDataset(dataset, indx, value)      # split on feature index and value
                  probability = len(subDataset)/float(len(dataset))
                  featEntropy += probability * calculateEntropy(subDataset) # sum Entropy for all feature vals
              infoGain = baseEntropy - featEntropy    # calculate the info gain; ie reduction in Entropy
              if infoGain > bestInfoGain:             # compare this to the best gain so far
                  bestInfoGain = infoGain             # if better than current best, set it to best
                  bestFeature = indx
          return bestFeature                          # return an best feature index


The code above is used to find the feature that can produce the highest information gain (across all its feature values) for the given set of labels. This means, that the choosen feature can split the data uniformly and produce the "purest" set of terminating nodes.

Initially the function calculates the baseEntropy for all labels of the dataset (which will be used to compare information gain). Then for each feature it calculates the featEntropy (feature Entropy) by dividing the dataset into various subgroup. It then calculates the sum of all subgroup Entropies for all feature values. In information theory, featEntropy is often referred to as the "information content of a message". In our case it is calculated using the following:

&nbsp; &nbsp; &nbsp; featEntrop = - Px<sub>0</sub> * log<sub>2</sub> (Px<sub>0</sub>) - Px<sub>1</sub> * log<sub>2</sub> (Px<sub>1</sub>)

### Choosing Root Node
The code listing below shows the featEntropy for each label group when given the full 7 data records. The calculation and spliting show how the node *non-surfacing* is choosen as the top root node of the tree. 

    Dataset: [[1, 1, 'yes'], [1, 1, 'yes'], 
              [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'], 
              [1, 1, 'maybe'], [0, 0, 'maybe']]

    labels: ['yes', 'yes', 'no', 'no', 'no', 'maybe', 'maybe']
    => baseEntropy = 1.5566567074628228
    
    indx=0; feature=non-surfacing
    subDatasets: [[0,1,'no'],[0,1,'no'],[0,0,'maybe']]   [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[1,1,'maybe']]
    labels: [['no', 'no', 'maybe'],['yes', 'yes', 'no', 'maybe']] 
    => featEntropy = 1.2506982145947811
    => infoGain0 =  0.3059584928680417
    
    indx=1; feature=flippers
    subDatasets: [[0,0,'maybe'],[1,0,'no']]   [[0,1,'no'],[0,1,'no'],[1,1,'yes'],[1,1,'yes'],[1,1,'maybe']]
    labels:  [['no', 'maybe'],['yes', 'yes', 'no', 'no', 'maybe']] 
    => featEntropy = 1.3728057820624016
    => infoGain1 =  0.1838509254004212
    
    infoGain0 > infoGain1
    =>  bestFeat=0  (on-surfacing) is the best feature to split and create the root node

When we look at the two subDataset groups for indx=0 we see that *non-surfacing* splits the labels more uniformly with all *yes* values in one subgroup and two *no* values in the other. Whereas the flippers feature creates two subDatasets that are less pure, ie. the first one contains a *no* and a *maybe* label and the second subDataset contains the other 5 labels. As a result, *non-surfacing* was choosen as the root node in the tree.

It is important to note the the choosen feature is not the one that creates the highest Entropy but rather the one that creates the least because information gain (purity) increases when Entropy decreases (because of the minus sign between baseEntropy and featEntropy).

## Part 3: Looping and Tree Building
 
This part is the heart of the Decision Tree algorithm. In each recursive iteration, a new node is choosen to branch the tree and then a new set of sub-datasets are passed to the next iteration. The algorithm has 8 steps:
  1- Start the algorithm with the a given dataset and a feature list
  2- Check two recursion terminating conditions
  3- Choose best feature to split the given dataset using Entropy test
  4- Make a copy of fetures values of best dataset & copy features list without best feature column 
  5- Create an empty tree node
  6- Split given dataset based on the best feature and its values
  7- Grow subtree for the empty tree node and append it to the main tree
  8- Iterate to step 6 with for the all given values for best feature

Here is the Python code for the 8 steps:

      def createTree(dataset, features):
          labels = [record[-1] for record in dataset]
          if labels.count(labels[0]) == len(labels): 
              return labels[0]            # stop splitting when all of the labels are equal

          if len(dataset[0]) == 1:        # stop splitting when there are no more features in dataset
              mjcount = max(labels,key=labels.count)
              return (mjcount) 

          bestFeat = chooseBestFeatureToSplit(dataset)
          bestFeatLabel = features[bestFeat]
          featValues = {record[bestFeat] for record in dataset}     # put feature values into a set
          subLabels = features[:]             # make a copy of features
          del(subLabels[bestFeat])            # remove bestFeature from labels list

          myTree = {bestFeatLabel:{}}         # value is empty dict
          for value in featValues:
              subDataset = splitDataset(dataset, bestFeat, value)
              subTree = createTree(subDataset, subLabels)
              myTree[bestFeatLabel].update({value: subTree})  # add (key,val) item into empty dict
          return myTree    
    
To understand this recursive function it is important to understand the subdataset being generated and used in the next recursive call and understand the two terminating conditions.

### Dataset Splitting
The following debugging output shows how at each level of the recursion a subdataset is passed to the function to generate the corresponding tree branch or leaf node.  

Here we show how the 2 left leaf nodes 'maybe' and 'no' under non-surfacing = False are generated. 

      Dataset: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'maybe'], [0, 0, 'maybe']]
      labels: ['yes', 'yes', 'no', 'no', 'no', 'maybe', 'maybe']
      features:  ['non-surfacing', 'flippers']
      best Feature to split: non-surfacing (0)

      sub-dataset: [[1, 'no'], [1, 'no'], [0, 'maybe']]
      labels: ['no', 'no', 'maybe']
      features:  ['flippers']
      best Feature to split: flippers (0)

      suf-dataset: [['maybe']]
      labels: ['maybe']      
      features:  []
      leaf node: maybe
      ===========

      sub-dataset: [['no'], ['no']]
      labels: ['no', 'no']      
      features:  []
      leaf node: no
      ===========

Here we show how the 2 right leaf nodes 'no' and 'yes' under non-surfacing = True are generated. 

      Dataset: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'maybe'], [0, 0, 'maybe']]
      labels: ['yes', 'yes', 'no', 'no', 'no', 'maybe', 'maybe']
      features:  ['non-surfacing', 'flippers']
      best Feature to split: non-surfacing (0)

      sub-dataset: [[1, 'yes'], [1, 'yes'], [0, 'no'], [1, 'maybe']]
      labels: ['yes', 'yes', 'no', 'maybe']
      features:  ['flippers']
      best Feature to split: flippers (0)

      sub-dataset: [['no']]
      labels: ['no']
      features:  []
      leaf node: no
      ===========

      sub-dataset: [['yes'], ['yes'], ['maybe']]
      labels: ['yes', 'yes', 'maybe']      
      features:  []
      leaf node: yes
      ===========
      

### Terminating Condition
Recursion terminates and a leaf node is generated in the decision tree when either of these two conditions is reached:
 
 1- All labels in the dataset have same value
 2- No more features in the feature list 

Let look back at the above dataset spliting to
The algorithm then calls itself recursively to do the pattern search, entropy test and spliting on the new sub-datasets. Recursion terminates and the tree branch is rolled when there are no more features to split in the sub-dataset or when all the prediction labels are the same.

## Concluding Remarks
We can clearly see now after this deep dive that the PageRank sample program that comes with Spark 2.0 looks deceivingly simple. The code is both compact and efficient. To understand how things actually work requires a deeper understanding of Spark RDDs, Spark's Scala based functional API, as well as Page Ranking formula. Programming in Spark 2.0 requires unraveling those RDDs that are implicitly generated on your behalf. 
