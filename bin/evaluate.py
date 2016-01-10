import os
import csv
import numpy
import sys; sys.path.append('.')
from dataset import *
from attribute import *
from axis_aligned_decision_tree import *
from pruned_decision_tree import *
from decision_forest import *
import datetime
import cProfile

def aa_decision_tree(dataset):
    return AxisAlignedDecisionTree(dataset, min_samples_split=10, max_depth=100)

def pruned_aa_decision_tree(dataset):
    return PrunedDecisionTree(dataset, tree_class=aa_decision_tree, split_fraction=0.75)

def parallel_forest(dataset):
    return DecisionForest(dataset, n_processes=1, n_trees=50, tree_class=aa_decision_tree)

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
    if dataset_file.startswith('hill'): continue
    if dataset_file.startswith('letter'): continue
    print("*******************\n"+dataset_file+"\n*******************")
    dataset = generate_dataset(os.path.join(dataset_path, dataset_file))
    compare([aa_decision_tree, pruned_aa_decision_tree, parallel_forest], dataset)
    print()
