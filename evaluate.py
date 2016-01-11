 #! /usr/bin/env python

import os
import csv
import numpy
from oo_trees.dataset import *
from oo_trees.attribute import *
from oo_trees.decision_tree import *
from oo_trees.random_forest import *
from oo_trees.classifier import *
import datetime
import sklearn.tree
import sklearn.ensemble

def aa_decision_tree(dataset):
    return DecisionTree(dataset, min_samples_split=10, max_depth=100)

def aa_random_forest(dataset):
    return RandomForest(dataset, n_trees=25, tree_class=aa_decision_tree)

class SklearnTree(Classifier):
    def __init__(self, dataset):
        self.tree = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
        self.tree.fit(dataset.X, dataset.y)

    def classify(self, x):
        return self.tree.predict([x])[0]

class SklearnForest(Classifier):
    def __init__(self, dataset):
        self.forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy')
        self.forest.fit(dataset.X, dataset.y)

    def classify(self, x):
        return self.forest.predict([x])[0]

def generate_dataset(filename):
    # Convert CSV to numpy arrays of X and y
    X = []
    y = []
    with open(filename, 'r') as csv_file:
         for row in csv.reader(csv_file):
             if row[0] == 'bOrdinal': continue
             X.append([float(n) for n in row[0:-1]])
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

def evaluate(classifier_class, training_dataset, test_dataset):
    t1 = datetime.datetime.now()
    classifier = classifier_class(training_dataset)
    t2 = datetime.datetime.now()
    print('took', t2-t1, 'to generate', classifier_class)
    performance = classifier.performance_on(test_dataset)
    print('accuracy of', classifier_class, 'was:')
    print(performance.accuracy)
    print(performance.to_array())

def compare(classifier_classes, dataset):
    training_dataset, test_dataset = dataset.random_split(0.75)
    for classifier_class in classifier_classes:
        evaluate(classifier_class, training_dataset, test_dataset)

dataset_path = './ccf/Datasets'
dataset_files = os.listdir(dataset_path)
for dataset_file in dataset_files:
    print("*******************\n"+dataset_file+"\n*******************")
    dataset = generate_dataset(os.path.join(dataset_path, dataset_file))
    compare([aa_decision_tree, SklearnTree, SklearnForest, aa_random_forest], dataset)
    print()
