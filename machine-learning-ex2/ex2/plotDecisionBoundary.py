import numpy as np
import matplotlib.pyplot as plt

from mapFeature import is_boundary

def plotDecisionBoundary(theta, X, y, data_X=None):
    """ PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    positive examples and o for the negative examples. X is assumed to be 
    a either 
    1) Mx3 matrix, where the first column is an all-ones column for the 
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """

    if X.shape[1] < 3:
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
        plt.show()
    else:
        def label_filter_factory(label):
            def label_filter(x):
                return x[1] == label 
            return label_filter

        pos = map(lambda x: x[0], filter(label_filter_factory(1), zip(data_X,y)))
        neg = map(lambda x: x[0], filter(label_filter_factory(0), zip(data_X,y)))
        p = plt.scatter(*zip(*pos))
        n = plt.scatter(*zip(*neg))
        plt.legend((p,n), ('y = 1', 'y = 0'))
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')

        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        xs = []
        ys = []
        for i in range(len(u)):
            for j in range(len(v)):
                if is_boundary(u[i], v[j]):
                    xs.append(u[i])
                    ys.append(v[i])
        # plt.plot(xs, ys)
        plt.scatter(xs, ys)
        plt.show()


