# MLP

###### 1. Initialize MLP

1. Set initial weights and biases for each layer.
2. A layer **l** with **n** neurons followed by a layer **l+1** with **m** neurons, has a weight's matrix **W<sub>l</sub>** of shape **nxm** and a bias' matrix **B<sub>l</sub>** of shape **1xn**.

###### 2. Forward propagation

1. Propagate batch of samples and store outputs of each layer before (**Z<sub>l</sub>**) and after (**Y<sub>l</sub>**) activation function. 
2. How?
- **n** = number of input neurons / **m** = number of samples in batch (batch size) / **k** = n° of neurons in layer l+1
* Weights (initial) --> W<sub>0</sub> = I<sub>n</sub> (Identity matrix)
* Inputs (first layer) --> X<sub>l</sub> = X<sub>mxn</sub>
* Biases (first layer) --> B<sub>l</sub> = 0<sub>1xn</sub>
* Outputs/Activations (first layer) --> Z<sub>l</sub> = Y<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>0</sub>) + B<sub>l</sub>
* Weights first layer --> W<sub>l</sub> = W<sub>nxk</sub>

For every other layer:
* Inputs --> X<sub>l</sub> = Y<sub>l-1</sub> 
* Biases --> B<sub>l</sub> = B<sub>1xm</sub> (m = n° of neurons in layer l)
* Output --> Z<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>l-1</sub>) + B<sub>l</sub>
* Activations --> Y<sub>l</sub> = ActivationFunction(Z<sub>l</sub>)
* Weights --> W<sub>l</sub> = W<sub>mxn</sub> (n = n° of neurons in layer l+1)

Note: the weights of a layer l refer to the connections between layer l and layer l+1, therefore, the output layer doesn't have a weight's matrix.


