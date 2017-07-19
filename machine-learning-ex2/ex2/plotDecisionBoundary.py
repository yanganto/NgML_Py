import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(theta, X, y):
    """ PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    positive examples and o for the negative examples. X is assumed to be 
    a either 
    1) Mx3 matrix, where the first column is an all-ones column for the 
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #
    def label_filter_factory(label):
        def label_filter(x):
            return x[1] == label 
        return label_filter

    pos = map(lambda x: x[0], filter(label_filter_factory(1), zip(X,y)))
    neg = map(lambda x: x[0], filter(label_filter_factory(0), zip(X,y)))
    p = plt.scatter(*zip(*pos))
    n = plt.scatter(*zip(*neg))
    plt.legend((p,n), ('Admitted', 'Not admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    x_axis = [e[0] for e in  X]
    xs = np.arange(min(x_axis), max(x_axis), 0.1)
    l = lambda x: (-1 / theta[2]) * (theta[1] * x + theta[0])
    vfunc = np.vectorize(l)
    ys = vfunc(xs)

    plt.plot(xs, ys)
    # =========================================================================

    plt.show()


