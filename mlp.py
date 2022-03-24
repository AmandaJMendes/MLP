#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:01:28 2022

@author: amanda
"""

import numpy as np

#For more on the derivatives of the activation functions, check derivativeActs.pdf

class Layer:
    """
    This is the model of a layer
    """
    def __init__(self, neurons):
        self.neurons = neurons #Number of neurons in this layer
        self.neurons_out = None #Number of neurons in the next layer
        self.weights = None #Weights between this layer and the next
        self.biases = None #Bias for each of the neurons in this layer
        self.outs = None #Outputs of this layer before activation function
        self.acts = None #Outputs of this layer after activation function
        self.deltas = None #Deltas for each of the neurons in this layer
        
    def init_weights(self):
        """
        This function initializes weights and bias
        """
        #Xavier initialization with a uniform distribution
        l = np.sqrt(6/(self.neurons+self.neurons_out))
        self.weights = np.random.uniform(-l,l, size = (self.neurons, self.neurons_out)) 
        #Bias initialized to 0
        self.biases = np.zeros((self.neurons)) 
     
    def activation(self, z):
        """
        This function returns f(z) where f is the activation function.
        """
        pass
    
    def derivative(self, x):
        """
        x = f(z), where f is the activaiton function.
        
        This function returns f'(z).
        """
        pass
        
    def forward(self, inputs, weights):
        """
        This function computes:
            -Weighted sum of inputs + bias: self.outs
            -Outputs (after activation function): self.acts 
        """
        self.outs = np.dot(inputs, weights) + self.biases
        self.acts = self.activation(self.outs)
        
    def backpropagate(self, deltas):
        """
        This function computes the deltas for each neuron of this layer given
        the deltas from the previous layer (backpropagation direction).
        """
        y = np.dot(deltas, self.weights.T)
        self.deltas = y * self.derivative(self.acts)        

class Identity(Layer):
    """
    This is an Identity layer
    """
    def __init__(self, neurons):
        Layer.__init__(self, neurons)
        
    def activation(self, x):
        """
        x = [z1, z2, ..., zn], where n is the number of neurons in the layer
        
        f(zi) = zi, where f is identity activation function and 1 <= i <= n
        f(x) = x
        
        This function returns f(x)
        """
        return x
    
    def derivative(self, x):
        """
        x = [y1, y2, ..., yn], where n is the number of neurons in the layer
        yi = f(zi), where f is identity activation function and 1 <= i <= n
        
        f'(zi) = 1 
        f'(x) = [f'(z1), f'(y2), ..., f'(yn)]
        
        This function returns f'(x)
        """
        return np.ones_like(x)
     
class Sigmoid(Layer):
    """
    This is a Sigmoid layer
    """
    def __init__(self, neurons):
        Layer.__init__(self, neurons)
        
    def activation(self, x):
        """
        x = [z1, z2, ..., zn], where n is the number of neurons in the layer
        
        f(zi) = 1/(1 + exp(-zi)), where f is sigmoid activation function and 1 <= i <= n        
        f(x) = [f(z1), f(z2), ..., f(zn)]
        
        This function returns f(x)
        """

        func = np.vectorize(lambda y: 1 / (1+ np.exp(-y)))
        return func(x)
    
    def derivative(self, x):
        """
        x = [y1, y2, ..., yn], where n is the number of neurons in the layer
        yi = f(zi), where f is sigmoid activation function and 1 <= i <= n
        f'(zi) = f(zi) * (1-f(zi))
        
        f'(zi) = yi * (1-yi)
        f'(x) = [f'(z1), f'(z2), ..., f'(zn)]
        
        This function returns f'(x)
        """
        return  x * (1 - x)
    
class ReLU(Layer):
    """
    This is a ReLU layer
    """
    def __init__(self, neurons):
        Layer.__init__(self, neurons)
        
    def activation(self, x):
        """
        x = [z1, z2, ..., zn], where n is the number of neurons in the layer
        
        f(zi) = max(0, zi), where f is ReLU activation function and 1 <= i <= n
        f(x) = [f(z1), f(z2), ..., f(zn)]
        
        This function returns f(x)
        """
        return np.clip(x, 0, None)
    
    def derivative(self, x):
        """
        x = [y1, y2, ..., yn], where n is the number of neurons in the layer
        yi = f(zi), where f is ReLU activation function and 1 <= i <= n
        f'(zi) = 1 (zi>0 --> yi>0)
               = 0 (zi<0 --> yi=0)
               = undefined (zi=0 --> yi=0)
              
        f'(zi) = 1 (yi>0) or 0 (yi=0)
        f'(x) = [f'(z1), f'(z2), ..., f'(zn)]
        
        This function returns f'(x)
        """
        ders = x.copy()
        ders[ders > 0] = 1
        return ders
                  
class Softmax(Layer):
    """
    This is a Softmax layer
    """
    def __init__(self, neurons):
        Layer.__init__(self, neurons)
        
    def activation(self, x):
        """
        x = [z1, z2, ..., zn], where n is the number of neurons in the layer
        sum = exp(z1)+exp(z2)+...+exp(zn)
        f(zi) = exp(zi)/sum, where f is softmax activation function and 1 <= i <= n
        f(x) = [f(z1), f(z2), ..., f(zn)]
        
        This function returns f(x)
        """
        r, c = x.shape
        vector = np.zeros((r, c))       
        for i in range(r):
            sum_i = np.sum(np.exp(x[i]))
            for j in range(c):
                vector[i][j] = np.exp(x[i][j])/sum_i           
        return vector
    
    def derivative(self, x):
        """
        x = [y1, y2, ..., yn], where n is the number of neurons in the layer
        yi = f(zi), where f is softmax activation function and 1 <= i <= n
        f'(zi) w.r.t. zk:
            i = k: f(zi)*(1-f(zi))
            i != k: -f(zi)*f(zk)
        
        f'(zi) w.r.t. zk:
            i = k: yi*(1-yi)
            i != k: -yi*yk
        f'(x) = [[f'(z1) wrt z1, f'(z1) wrt z2, ..., f'(z1) wrt zn],
                 [f'(z2) wrt z1, f'(z2) wrt z2, ..., f'(z2) wrt zn],
                 [................................................],
                 [f'(zn) wrt z1, f'(zn) wrt z2, ..., f'(zn) wrt zn]]
        
        This function returns f'(x)
        """
        samples, neurons = x.shape
        derivs = np.zeros((samples, neurons, neurons))
        for n in range(samples):
            for i in range(neurons):
                for j in range(neurons):
                    if i == j:
                        derivs[n][i][j] =  x[n][i] * (1-x[n][i])
                    else:
                        derivs[n][i][j] = - x[n][i] * x[n][j]      
        return derivs
    
    def backpropagate(self, deltas):
        """
        Backpropagation for a softmax layer is different because f(zi), where f
        is softmax activation function, depends on all the neurons of the layer,
        not only on neuron i. Therefore, the derivative of softmax for a given
        neuron is actually an array of partial derivatives.
        """
        y = np.dot(deltas, self.weights.T)
        dd = []
        derivatives = self.derivative(self.acts) 
        for i in range(y.shape[0]):
            dd.append(np.dot(y[i], derivatives[i]))
        self.deltas = np.array(dd)              
        
def Input(neurons):
    """
    This function creates a input layer.
    The input layer is always an Identity Layer.
    """
    return Identity(neurons)

def Hidden(activation, neurons):
    """
    This function creates a hidden layer according to the desired
    activation function.
    """
    return activation(neurons)

def Output(activation, neurons):
    """
    This function creates an output layer.
    
    The output layer inherits the properties from the layer of the
    desired activation function.
    """
    class OutputLayer(activation):
        def __init__(self, neurons):
            activation.__init__(self, neurons)
            
        def init_weights(self):
            """
            The output layer doesn't have weights connecting it to the next layer.
            """
            self.biases = np.zeros((self.neurons))
            
        def backpropagate(self, errors):
            """
            The backpropagation in the output layer is different from the 
            backpropagation in the hidden layers.
            
            For de output layer, the deltas are computed based on the derivatives
            of the loss function.
            """
            if activation == Softmax:
                dd = []
                derivatives = self.derivative(self.acts) 
                for i in range(errors.shape[0]):
                    dd.append(np.dot(errors[i], derivatives[i]))
                self.deltas = np.array(dd)                
            else:
                self.deltas = errors * self.derivative(self.acts)
            
    return OutputLayer(neurons)

           
class MLP:
    """
    This is a Multilayer perceptron
    """
    def __init__(self):
        self.layers = [] #List of all the layers
        self.n_layers = None #Number of layers
        
    def add_layer(self, *args):
        """
        This function adds layers to the MLP
        """
        for l in args:
            self.layers.append(l)
        
    def compile_mlp(self):
        """
        This function compiles the MLP:
            -Sets the neurons_out attribute for every layer but the output layer
            -Initializes weights for all the layers
        """
        self.n_layers = len(self.layers)
        for i in range(len(self.layers)):
            if i != self.n_layers - 1: 
                self.layers[i].neurons_out = self.layers[i+1].neurons    
            self.layers[i].init_weights()
            
    def propagate(self, x):
        """
        This function propagtes input x through the MLP
        """
        weights = np.eye(self.layers[0].neurons) #There are no weights before the input layer,
                                                 #so the weights are initially an identity matrix
        for l in range(self.n_layers):
            self.layers[l].forward(x, weights)
            if l != self.n_layers - 1: #The output layer doesn't have weights
                weights = self.layers[l].weights
                x = self.layers[l].acts
    
    def evaluate(self, x, y, metric):
        """
        This function uses a metric to evaluate the performance of the MLP 
        on (x, y) set.
        """
        self.propagate(x) #The input x is propagated
        if type(metric) == type: #If the metric is a class, the metric is actually
                                 #the loss function within the class
            metric = metric().loss
        return metric(y, self.layers[-1].acts) #Evaluate the model given the
                                               #actual output and the predicted output
    
    def predict(self, x):
        """
        This function predicts the output for an input x
        """
        self.propagate(x)
        return self.layers[-1].acts
    
    def backpropagate(self, errors):
        """
        This function backpropagates the derivatives of the loss function
        (errors) through the MLP
        """
        for i in range(self.n_layers-1, 0, -1): #The backpropagation goes from
                                                #the output layer to the input layer
            l = self.layers[i]
            l.backpropagate(errors)
            errors = l.deltas

    def optimize(self, optimizer):
        """
        This function updates the weights in the MLP
        """
        for i in range(self.n_layers-1): #For every layer in the MLP
            l = self.layers[i]
            n = l.acts.shape[0]
            gradients = []
            for s in range(n): #For every sample, compute the gradients 
                               #for all the weights between this layer and the next one
                gradients.append(self.layers[i].acts[s].reshape(-1,1) * self.layers[i+1].deltas[s])
                
            #The weights and bias are updated considering the average of the gradients for all the samples
            #The gradients for the bias are equal to the deltas
            l.weights -= optimizer.delta_ws(np.mean(gradients, axis=0)) 
            self.layers[i+1].biases -= optimizer.delta_bs(np.mean(self.layers[i+1].deltas, axis=0))
            
    def train (self, x, y, loss, optimizer, epochs, batch_size = 1, show = False):
        """
        This function trains de MLP through multiple iterations (epochs) of
        forward and backpropagation on a (x, y) set. 
        """
        for e in range(epochs):
            for i in range(int(np.ceil(len(x)/batch_size))): #The samples are passed to MLP 
                                                             #in groups of size = batch_size
                first = i*batch_size
                last = min(first+batch_size, len(x)) #The last batch has the
                                                     #remaining samples so its size
                                                     #may not be equal to batch_size. 
                self.propagate(x[first:last]) #Forward propagation
                if show:
                    print ("Loss: ", loss.loss(y[first:last], self.layers[-1].acts))
                self.backpropagate(loss.derivative(y[first:last], self.layers[-1].acts)) #Backpropagation
                self.optimize(optimizer) #Weight's update
                
                
