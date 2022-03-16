# MLP

###### 1. Initialize MLP

1. Set initial weights and biases for each layer.
2. A layer **l** with **n** neurons followed by a layer **l+1** with **m** neurons, has a weight's matrix **W<sub>l</sub>** of shape **nxm** and a bias' matrix **B<sub>l</sub>**

###### 2. Forward propagation

1. Propagate batch of samples and store outputs of each layer before (**Z<sub>l</sub>**) and after (**Y<sub>l</sub>**) activation function. 
2. How?
* Weights first layer --> W<sub>l</sub> = I<sub>n</sub> (Identity matrix, n = number of input neurons)
* Inputs first layer --> X<sub>l</sub> = X<sub>mxn</sub> (m = number of samples in batch (batch size))
* Outputs/Activations first layer --> Z<sub>l</sub> = Y<sub>l</sub> = X<sub>l</sub> \cdot 
