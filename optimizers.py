#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:41:22 2022

@author: amanda
"""

"""
Ideas for other optimizers:
    -SGD with momentum
    -NAG
    -RMSprop
    -Adam
"""
class SGD:
    """
    This is a SGD optimizer
    """
    def __init__(self, learning_rate):
        self.lr = learning_rate
        
    def delta_ws(self, gradients):
        """
        This function returns the change required for every weight given the
        gradient of each weight w.r.t. the loss and a learning rate.
        """
        return self.lr * gradients
    
    def delta_bs(self, deltas):
        """
        This function returns the change required for every bias given the
        gradient of each weight w.r.t. the loss and a learning rate.
        """
        return self.lr * deltas
