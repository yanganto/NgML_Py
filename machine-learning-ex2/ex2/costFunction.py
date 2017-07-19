import numpy as np

from sigmoid import sigmoid

def costFunction(theta, X, y, cost_only=False, grad_only=False):
    """ COSTFUNCTION Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    """


    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    if not grad_only:
        J = np.sum( -1 * np.dot(y, np.log(sigmoid(np.matmul(X, theta)))) - np.dot((1 - y), np.log(1 - sigmoid(np.matmul(X, theta)))) ) / m 

    if cost_only:
        return J

    if grad_only:
        z = np.dot(X,theta)            
        h = sigmoid(z);
    
        grad = (float(1)/m)*((h-y).T.dot(X))          
        return grad

    grad = np.zeros(theta.shape)
    tmp = np.zeros(theta.shape)
    for j in range(X.shape[1]): 
        tmp[j] = np.sum(np.matmul(np.array([X[:, j]]), sigmoid(np.matmul(X, theta)) - y.reshape(m,1))) / m
    grad = tmp
    # =============================================================
    return J, grad
