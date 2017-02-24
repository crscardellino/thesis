Experiments Readme
==================

This is a readme to keep track of the experiments names.

Experiments
-----------

### Experiment 0

+ Has the basic info to gather all the important metrics to see what are the
  best classifiers for the tasks.
+ It is also useful to select between representations.
+ The classifiers are:
    - Baseline
    - Decision Tree with Gini impurity
    - Logistic Regression
    - Multilayer Perceptron with a hidden layer of 5000 units
    - Naive Bayes
    - SVM with linear kernel
+ The representations are:
    - Handcrafted features selecting the best 10000 using scikit learn
      SelectKBest
    - Hashed positive 10000 features
    - Hashed negative 10000 features: this representation is not valid for
      Naive Bayes

#### Experiment 0 bis

+ Same as before, but the handcrafted features are not cut to 10000 but use all
  instead. Is done to see if there's a real difference between selecting
  features or not. It is not used with the MLP as it won't fit in memory.

### Experiment 1

+ Once the best representation and classifiers are selected, we use them to
  explore the folds experiment to prove the first subhypothesis of supervised
  learning chapter.
