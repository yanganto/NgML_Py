""" Machine Learning Online Class - Exercise 2: Logistic Regression
 
   Instructions
   ------------
  
   This file contains code that helps you get started on the logistic
   regression exercise. You will need to complete the following functions 
   in this exericse:
 
      sigmoid.py
      costFunction.py
      predict.py
      costFunctionReg.py
 
   For this exercise, you will not need to change any code in this file,
   or any other files other than those mentioned above.
 
"""
from functools import partial

import numpy as np
from scipy import optimize

from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from costFunction import costFunction
from sigmoid import sigmoid
from predict import predict

## Initialization

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.genfromtxt('ex2data1.txt', delimiter=',')
data_X = data[:, (0, 1)]; y = data[:, 2];

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(data_X, y)

print('\nProgram paused. Press enter to continue.\n')


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = data_X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m,1)), data_X), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))


# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): %f\n'% cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost, grad = costFunction(test_theta, X, y)

print('\nCost at test theta: %f\n' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

print('\nProgram paused. Press enter to continue.\n')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

# #  Set options for fminunc
# options = optimset('GradObj', 'on', 'MaxIter', 400);

# #  Run fminunc to obtain the optimal theta
# #  This function will return theta and the cost 
# [theta, cost] = ...
# 	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
r= optimize.fmin_bfgs(partial(costFunction, cost_only=True), initial_theta, 
        fprime=partial(costFunction, grad_only=True), args=(X, y), maxiter=400,
        full_output = True)
# # Print theta to screen
print('Cost at theta found by fminunc: %f\n' % r[1]);
print('Expected cost (approx): 0.203\n');
print('theta: \n');
print(r[0]);
print('Expected theta (approx):\n');
print(' -25.161\n 0.206\n 0.201\n');

# Plot Boundary
plotDecisionBoundary(r[0], data_X, y)

print('\nProgram paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.matmul(np.array([1, 45, 85]), r[0]))
print('For a student with scores 45 and 85, we predict an admission probability of %F\n' % prob);
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(r[0], X)

validation = np.logical_and(p, y.reshape(m,1))
print('Train Accuracy: %f\n' % (np.count_nonzero(validation) * 100 / m))
print('Expected accuracy (approx): 89.0\n')
print('\n')


