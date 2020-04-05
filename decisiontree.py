#!/usr/bin/env python3
"""
Created on Fri Mar 20 17:42:29 2020
@author: abbas taher

Based on:
   Decision Tree Source Code for Machine Learning in Action Ch. 3
   @author: Peter Harrington


Output
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
 
 maybe,no,no,yes,
"""

from math import log
from collections import defaultdict
import json
import pprint


def calculateEntropy(dataset):
    counter= defaultdict(int)   # number of unique labels and their frequency
    for record in dataset:      
        label = record[-1]      # always assuming last column is the label column 
        counter[label] += 1
    entropy = 0.0
    for key in counter:
        probability = counter[key]/len(dataset)           # len(dataSet) = number of entries   
        entropy -= probability * log(probability,2)       # log base 2
    return entropy

def splitDataset(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    baseEntropy = calculateEntropy(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    
    numFeat = len(dataset[0]) - 1          # do not include last label column     
    for indx in range(numFeat):            # iterate over all the features index
        featValues = {record[indx] for record in dataset}     # put feature values into a set
        featEntropy = 0.0
        for value in featValues:
            subDataset = splitDataset(dataset, indx, value)      # split based on feature index and value
            probability = len(subDataset)/float(len(dataset))
            featEntropy += probability * calculateEntropy(subDataset) # sum Entropy for all feature values

        infoGain = baseEntropy - featEntropy    # calculate the info gain; ie reduction in Entropy
        if infoGain > bestInfoGain:             # compare this to the best gain so far
            bestInfoGain = infoGain             # if better than current best, set it to best
            bestFeature = indx
    return bestFeature                          # return an best feature index


def createTree(dataset, features):
    labels = [record[-1] for record in dataset]
    
    # Terminating condition #1
    if labels.count(labels[0]) == len(labels):   # stop splitting when all of the labels are same
        return labels[0]            
    # Terminating condition #2
    if len(dataset[0]) == 1:                     # stop splitting when there are no more features in dataset
        mjcount = max(labels,key=labels.count)   # select majority count
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


def predict(inputTree, features, testVec):
    
    def classify (inputTree, testDict):
        (key, subtree), = inputTree.items()
        testValue = testDict.pop(key)
        if len(testDict) == 0:
            return subtree[testValue]
        else:
            return classify(subtree[testValue], testDict)
            
    testDict = dict(zip(features, testVec))
    return classify(inputTree, testDict)
    

def pprintTree(tree):
    pprint.pprint (tree)
    tree_str = json.dumps(tree, indent=4)
    tree_str = tree_str.replace("\n    ", "\n")
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace(',', "")
    tree_str = tree_str.replace("{", "")
    tree_str = tree_str.replace("}", "")
    tree_str = tree_str.replace("    ", " | ")
    tree_str = tree_str.replace("  ", " ")    
    print (tree_str)


def createDataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'],
               [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'],
               [1, 1, 'maybe'], [0, 0, 'maybe']]
    
    features = ['non-surfacing','flippers']
    label = ['isfish']
    return dataset, features

    
def main():
    dataset, features = createDataset()
    tree = createTree(dataset, features)
    pprintTree (tree) 
    
    testVectors = [(0,0), (0,1),(1,0),(1,1)]
    for vec in testVectors:
        pred = predict(tree, features, vec)
        print (pred, end =',')


if __name__ == "__main__":
    main()
