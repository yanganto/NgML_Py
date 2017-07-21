"""
   Machine Learning Online Class - Exercise 2: Logistic Regression
 
   Instructions
   ------------
 
   This file contains code that helps you get started on the second part
   of the exercise which covers regularization with logistic regression.
 
   You will need to complete the following functions in this exericse:
 
      sigmoid.py
      costFunction.py
      predict.py
      costFunctionReg.py
 
   For this exercise, you will not need to change any code in this file,
   or any other files other than those mentioned above.
"""
 

## Initialization
from functools import partial

import numpy as np
from scipy import optimize

from costFunctionReg import costFunctionReg
from mapFeature import mapFeature
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.genfromtxt('ex2data2.txt', delimiter=',')
data_X = data[:, (0, 1)]; y = data[:, 2];
m, _ = data_X.shape

plotData(data_X, y, legend=('y = 1', 'y = 0'), xlabel='Microchip Test 1', ylabel='Microchip Test 2')


## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(data_X[:, 0], data_X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
lambda_parameter = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_parameter)

print('Cost at initial theta (zeros): %f\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

print('\nProgram paused. Press enter to continue.\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],1))
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): %f\n'% cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

print('\nProgram paused. Press enter to continue.\n')

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_parameter = 1

# Optimize
r = optimize.fmin_bfgs(partial(costFunctionReg, cost_only=True), initial_theta, 
        fprime=partial(costFunctionReg, grad_only=True), args=(X, y, lambda_parameter),
        maxiter=400, full_output = True)

# Plot Boundary
plotDecisionBoundary(r[0], X, y, data_X)

# Compute accuracy on our training set
p = predict(r[0], X)
validation = np.logical_and(p, y.reshape(m,1))
print('Train Accuracy: %f\n' % (np.count_nonzero(validation) * 100 / m))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')

