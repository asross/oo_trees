import os
import csv
import numpy
import sys
sys.path.append('.')
from dataset import *
from attribute import *
from axis_aligned_decision_tree import *

def evaluate(filename, classifier_class):
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
    print performance.accuracy
    print performance.to_array()

evaluate('./ccf/Datasets/soybean.csv', AxisAlignedDecisionTree)
