import numpy as np

def accuracy(actual, estimated):
    """
    This function returns the accuracy given the actual and the estimated outputs.
    """
    eq = np.equal(np.argmax(actual, axis=1),
                  np.argmax(estimated, axis=1))
    
    return np.count_nonzero(eq)/len(actual)
