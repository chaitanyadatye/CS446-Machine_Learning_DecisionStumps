#!/bin/bash

mkdir bin
rm -rf 2.c.pred
rm -rf 2.d.4.pred
rm -rf 2.d.8.pred
make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../data/badges.train ./../badges.train.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../data/badges.test.blind ./../badges.test.arff

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
java -cp lib/weka.jar:bin cs446.homework2.WekaTesterTree ./../badges.train.arff ./../badges.test.arff ../data/badges.test.blind 2.c.pred -1
java -cp lib/weka.jar:bin cs446.homework2.WekaTesterTree ./../badges.train.arff ./../badges.test.arff ../data/badges.test.blind 2.d.4.pred 4
java -cp lib/weka.jar:bin cs446.homework2.WekaTesterTree ./../badges.train.arff ./../badges.test.arff ../data/badges.test.blind 2.d.8.pred 8
