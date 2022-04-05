#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:33:30 2022

@author: amanda
"""

import numpy as np

"""
Ideas for other loss functions:
    -Binary Cross-Entropy
    -RMSE

For more on the derivatives of the loss functions, check derivativesLoss.pdf
"""

class CrossEntropy:
    """
    This is a CrossEntropy loss function.
    """
    def __init__(self):
        pass
    
    def loss(self, actual, estimated):
        """
        This function returns the categorical crossentropy loss
        for the estimated outputs given the actual outputs.
        """
        offset = pow(10, -15)
        clipped = np.clip(estimated, offset, 1-offset) 
        return -np.sum(actual * np.log(clipped))/actual.shape[0] 
    
    def derivative(self, actual, estimated):
        """
        This function returns the derivatives of the crossentropy loss given
        the actual and estimated outputs.
        """
        return -actual/estimated

        
class MSE:
    """
    This is a Mean Squared Error function.
    """
    def __init__(self):
        pass
    
    def loss(self, actual, estimated):
        """
        This function returns the MSE for the estimated outputs
        given the actual outputs.
        """
        return np.sum((actual - estimated)**2)/actual.size
        
    def derivative(self, actual, estimated):
        """
        This function returns the derivatives of MSE given the actual and
        estimated outputs.
        """
        return 2*(estimated - actual)
    
