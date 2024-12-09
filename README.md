# MLFromScratch

This repository is intended to accelerate learning on foundational aspects of ML algorithms

Algorithms from 1 to 10 were implemented alongside [this playlist](https://youtube.com/playlist?list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&si=RGWpGwS_hmT4L-Ax) by [AssemblyAI](https://www.assemblyai.com/).

Algorithms from 11 onwards were implemented alongside Andrej Karpathy's [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=95qzGG_omB2Zulhf) playlist.

## 1 - KNN
A simple supervised learning algorithm that classifies data points based on the majority class of their k nearest neighbors in feature space. It's often used for classification tasks. 

[Code](./src/01%20-%20KNN/)

## 2 - Linear Regression
A supervised learning algorithm used for regression tasks. It models the relationship between dependent and independent variables by fitting a straight line (or hyperplane) to the data.

[Code](./src/02%20-%20Linear%20Regression/)

## 3 - Logistic Regression
A supervised learning algorithm for binary or multi-class classification problems. It uses a logistic (sigmoid) function to estimate probabilities of class membership.

[Code](./src/03%20-%20Logistic%20Regression/)

## 4 - Decision Tree
A tree-like structure used for both classification and regression. Each node represents a decision rule on a feature, and the leaves represent outcomes or predictions. It's intuitive but prone to overfitting.

[Code](./src/04%20-%20Decision%20Tree/)

## 5 - Random Forest
An ensemble learning method that uses multiple decision trees to improve prediction accuracy and reduce overfitting. Predictions are made by averaging (regression) or voting (classification) the results from all trees.

[Code](./src/05%20-%20Random%20Forest/)

## 6 - Naive Bayes
A probabilistic classifier based on Bayes' theorem with the assumption that features are conditionally independent given the class. It's fast and effective, especially for text classification.

[Code](./src/06%20-%20Naive%20Bayes/)

## 7 - PCA
An unsupervised dimensionality reduction technique that transforms the data into a set of orthogonal components, maximizing variance while reducing redundancy.

[Code](./src/07%20-%20PCA/)

## 8 - Perceptron
A basic neural network unit and one of the earliest models for binary classification. It computes a weighted sum of inputs, applies a step function, and updates weights using a simple learning rule.

[Code](./src/08%20-%20Perceptron/)

## 9 - Support Vector Machine (SVM)
A supervised learning algorithm for classification and regression. It finds the hyperplane that maximizes the margin between different classes in the feature space, often with the help of kernel functions.

[Code](./src/09%20-%20SVM/)

## 10 - KMeans
An unsupervised clustering algorithm that partitions data into k clusters by minimizing the variance within each cluster. Each cluster is defined by its centroid.

[Code](./src/10%20-%20KMeans/)

## 11 - micrograd
A small library designed for building and training neural networks with automatic differentiation. It focuses on understanding how backpropagation works by implementing gradients at a low level.

[Code](./src/11%20-%20micrograd/)

## 12 - makemore
A simple neural network-based text generation framework, often implemented as a character-level language model. It predicts the next character in a sequence to "make more" text.

[Code](./src/12%20-%20makemore/)

## 13 - GPT
A deep learning model based on the Transformer architecture, pre-trained on large amounts of text data for language understanding and generation tasks. It's widely used for tasks like text completion, summarization, and dialogue.

[Code](./src/13%20-%20GPT/)
