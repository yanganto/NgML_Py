import numpy as np

from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """ GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    m = len(y); # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        tmp1 = theta[0] -  alpha / m * np.sum( np.dot((np.matmul(X, theta) - y).ravel(), X[:, 0]))
        tmp2 = theta[1] -  alpha / m * np.sum( np.dot((np.matmul(X, theta) - y).ravel(), X[:, 1]))
        theta[0] = tmp1
        theta[1] = tmp2

        # ============================================================

        # Save the cost J in every iteration    
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
