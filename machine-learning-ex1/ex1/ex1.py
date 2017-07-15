"""
 Machine Learning Online Class - Exercise 1: Linear Regression Python version

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions
  in this exericse:

     warmUpExercise.py
     plotData.py
     gradientDescent.py
     computeCost.py
     gradientDescentMulti.py
     computeCostMulti.py
     featureNormalize.py
     normalEqn.py

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.

 x refers to the population size in 10,000s
 y refers to the profit in $10,000s
"""


## Initialization
import numpy as np

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent


## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')

print(warmUpExercise())

print('Program paused. Press enter to continue.')


## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y); # number of training examples

# 1D array to column vector
data_X = np.reshape(X, (m,1)) 
y = np.reshape(y, (m,1)) 

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(data_X, y)

print('Program paused. Press enter to continue.')

## =================== Part 3: Cost and Gradient descent ===================

#X = [np.ones(m, 1), data(:,1)]; # Add a column of ones to x
X = np.concatenate((np.ones((m,1)), data_X), axis=1)
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [[0], [0]]\nCost computed = ', J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1],[2]]))
print('\nWith theta = [ [-1], [2] ]\nCost computed = ', J)
print('Expected cost value (approx) 54.24');

print('Program paused. Press enter to continue.');

print('\nRunning Gradient Descent ...')
# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent:');
print('#f', theta);
print('Expected theta values (approx)');
print(' -3.6303\n  1.1664\n');

# Plot the linear fit
plotData(data_X, y, theta)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul(np.array([1, 3.5]), theta)[0]
print('For population = 35,000, we predict a profit of %f' % (predict1*10000) ) 
predict2 = np.matmul(np.array([1, 7]), theta)[0]
print('For population = 70,000, we predict a profit of %f' % (predict2*10000) )

print('Program paused. Press enter to continue.')

### ============= Part 4: Visualizing J(theta_0, theta_1) =============
#fprintf('Visualizing J(theta_0, theta_1) ...\n')
#
## Grid over which we will calculate J
#theta0_vals = linspace(-10, 10, 100);
#theta1_vals = linspace(-1, 4, 100);
#
## initialize J_vals to a matrix of 0's
#J_vals = zeros(length(theta0_vals), length(theta1_vals));
#
## Fill out J_vals
#for i = 1:length(theta0_vals)
#    for j = 1:length(theta1_vals)
#	  t = [theta0_vals(i); theta1_vals(j)];
#	  J_vals(i,j) = computeCost(X, y, t);
#    end
#end
#
#
## Because of the way meshgrids work in the surf command, we need to
## transpose J_vals before calling surf, or else the axes will be flipped
#J_vals = J_vals';
## Surface plot
#figure;
#surf(theta0_vals, theta1_vals, J_vals)
#xlabel('\theta_0'); ylabel('\theta_1');
#
## Contour plot
#figure;
## Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
#contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
#xlabel('\theta_0'); ylabel('\theta_1');
#hold on;
#plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
