import os
import csv
import numpy
import sys; sys.path.append('.')
from dataset import *
from attribute import *
from axis_aligned_decision_tree import *
from decision_forest import *
import datetime

def parallel_forest(dataset):
    return DecisionForest(dataset, n_processes=10)

def aa_decision_tree(dataset):
    return AxisAlignedDecisionTree(dataset, min_samples_split=10, max_depth=100)

def generate_dataset(filename):
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

    return Dataset(X, y, attributes)

import cProfile
def evaluate(classifier_class, training_dataset, test_dataset):
    print 'profiling generation of ', classifier_class
    cProfile.run("classifier = AxisAlignedDecisionTree(training_dataset)")
    performance = classifier.performance_on(test_dataset)
    print 'accuracy of', classifier_class, 'was:'
    print performance.accuracy
    print performance.to_array()

def compare(classifier_class1, classifier_class2, dataset):
    training_dataset, test_dataset = dataset.training_test_split(0.75)
    evaluate(classifier_class1, training_dataset, test_dataset)
    evaluate(classifier_class2, training_dataset, test_dataset)

dataset_path = './ccf/Datasets'
dataset_files = os.listdir(dataset_path)
for dataset_file in dataset_files:
    print dataset_file
    dataset = generate_dataset(os.path.join(dataset_path, dataset_file))
    training_dataset, test_dataset = dataset.random_split(0.75)
    evaluate(aa_decision_tree, training_dataset, test_dataset)
    print ""
