# Basic-Machine-Learning-

In this Repository, i have made codes for Linear regression and Logistic Regression.

In linear regression, Dataset used had 1 feature and corresponding output.

For logistic regression i had 2 datasets, one was linearly separable and other had non linear boundary, use can see all that in codes.

# Linear Regression
Suppose you have a dataset with some features and and an output(that can be predicted based on given features), here i have considered only linear regression so i'll get a straight line, fitting that dataset and later we can use that straight line to predict output for new values of features.
Below i will explain linear regression in simple steps:
1. Get training dataset say (X,y), where X is be matrix of (m,k) and y is a vector of size (m,1).....m is number of training examples and k is number of features we have., here k = 1
2. define a hypothesis function say : y_pred = aX + b, where a is an array of length k and b is an integer.
3. now we want y_pred to be close to y, so we define a cost function that will penalize y_pred for going away from y
4. various cost functions are available, but the one we use with linear regression is Mean squared error : (y_pred - y)^2 / m
5. now we start iterating, and want to minimize cost function varying a and b
6. for doing 5 we use a optimizer, again various optimisers are available, but we will use gradient descent.

These are the overall steps behind Linear regression, however there is inbuild command in scipy and sklearn, i have used scipy to do all this.


# Logistic Regression
Logistic regression is used for classification task, your training data will have some features and corresponding label(either 0 or 1).
Below i will explain logistic regression in simple steps:
1. Get training dataset say (X,y), where X is be matrix of (m,k) and y is a vector of size (m,1).....m is number of training examples and k is number of features we have., here k = 1
2. Define Hypothesis function : y_probability = g( aX + b ), where g(z) = 1/ (1 + e^(-z)). this gives out probabilty of each sample of X, and we consider y_probability < 0.5 to be considered negative sample and y_probability >= 0.5 to be positive sample
3. now we want y_pred to be close to y, so we define a cost function that will penalize y_pred for going away from y
4. various cost functions are available, but the one we use with logistic regression is Cross entropy loss
5. now we start iterating, and want to minimize cost function varying a and b
6. for doing 5 we use a optimizer, again various optimisers are available, but we will use gradient descent.

These are the overall steps behind Logistic regression, however there is inbuild command in  sklearn, i have used it to do all this.

The dataset i used in all 3 codes are picked from 
[Linear Regression](https://www.coursera.org/learn/machine-learning/programming/8f3qT/linear-regression). and 
[Logistic Regression](https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression).
