import os
import csv
import numpy
import sys; sys.path.append('.')
from dataset import *
from attribute import *
from axis_aligned_decision_tree import *
from decision_forest import *
import datetime

def evaluate(filename, classifier_class):
    t = datetime.datetime.now()

    # Convert CSV to numpy arrays of X and y
    X = []
    y = []
    with open(filename, 'rb') as csv_file:
         for row in csv.reader(csv_file):
             if row[0] == 'bOrdinal': continue
             X.append(map(float, row[0:-1]))
             y.append(float(row[-1]))
    X = numpy.array(X)
    y = numpy.array(y)

    # Figure out whether the attributes are categorical or numeric
    attributes = []
    for i in range(X.shape[1]):
        all_values = X[:, i]
        unique_values = numpy.unique(all_values)
        if len(unique_values) < 10 and len(all_values) > 25:
            attributes.append(CategoricalAttribute(i))
        else:
            attributes.append(NumericAttribute(i))

    # initialize the dataset
    dataset = Dataset(X, y, attributes)
    training_dataset, test_dataset = dataset.training_test_split(0.7)

    classifier = classifier_class(training_dataset)
    performance = classifier.performance_on(test_dataset)
    time = datetime.datetime.now() - t
    print 'done in ', time, ' seconds'
    print 'for ', filename.split('/')[-1], ' accuracy of ', classifier_class, ' was:'
    print performance.accuracy
    print performance.to_array()

evaluate('./ccf/Datasets/soybean.csv', AxisAlignedDecisionTree)

def forest(dataset): return DecisionForest(dataset, n_processes=20)
evaluate('./ccf/Datasets/soybean.csv', forest)
