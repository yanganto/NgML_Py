import numpy as np
def mapFeature(X1, X2):
    """ MAPFEATURE Feature mapping function to polynomial features
     
        MAPFEATURE(X1, X2) maps the two input features
        to quadratic features used in the regularization exercise.
     
        Returns a new feature array with more features, comprising of 
        X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
     
        Inputs X1, X2 must be the same size
     
    """

    # degree = 6
    out = np.zeros((X1.shape[0], 28))
    for r in range(X1.shape[0]):
        x1 = X1[r]
        x2 = X2[r]

        out[r,:] = polyNomal(x1, x2)
    return out

def polyNomal(x1, x2):
    return [1,
            x1, x2, 
            x1 **2, x1 * x2, x2 **2, 
            x1 **3, x1 **2 * x2, x1 * x2 **2, x2 ** 3,
            x1 **4, x1 **3 * x2, x1 **2 * x2 **2, x1 * x2 **3, x2 **4,
            x1 **5, x1 **4 * x2, x1 **3 * x2 **2, x1 **2 * x2 **3, x1 * x2 **4, x2 **5,
            x1 **6, x1 **5 * x2, x1 **4 * x2 **3, x1 **3 * x2 **3, x1**2 * x2 **4, x1 * x2 **5, x2 **6 ]

def is_boundary(x1, x2, error_level=11):
    return -error_level < sum(polyNomal(x1, x2)) < error_level 
