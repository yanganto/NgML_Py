import numpy as np

from sigmoid import sigmoid

def costFunctionReg(theta, X, y, lambda_parameter, cost_only=False, grad_only=False):
    """ COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    """


    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly 
    J = 0;
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    l = np.ones((1, theta.shape[0]))
    l[0] = 0

    if not grad_only:
        J = np.sum( -1 * np.dot(y, np.log(sigmoid(np.matmul(X, theta)))) - np.dot((1 - y), np.log(1 - sigmoid(np.matmul(X, theta)))) ) / m  + lambda_parameter * np.dot(l, (theta ** 2)) / (2 * m)

    if not cost_only:
        tmp = np.zeros(theta.shape)
        for j in range(theta.shape[0]):
            tmp[j] = np.dot((sigmoid(np.matmul(X, theta)).T- y), X[:, j]) / m
            if j:
                tmp[j] += lambda_parameter * theta[j] / m 
        grad = tmp

    # =============================================================
    if cost_only:
        return J

    if grad_only:
        return grad

    return J, grad
