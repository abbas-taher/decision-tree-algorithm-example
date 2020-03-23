def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               [0, 1, 'maybe'],
               [1, 0, 'maybe']]
    
    features = ['non-surfacing','flippers']
    label = ['isfish']
    
    return dataSet, features
