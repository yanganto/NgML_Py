import matplotlib.pyplot as plt
def plotData(X, y, legend=None, xlabel=None, ylabel=None):
    """ PLOTDATA Plots the data points X and y into a new figure 
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
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
    if legend:
        plt.legend((p,n), legend)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # =========================================================================

    plt.show()


