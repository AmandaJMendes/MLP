# MLP

###### 1. Initialize MLP

1. Set initial weights and biases for each layer.
2. A layer **l** with **n** neurons followed by a layer **l+1** with **m** neurons, has a weight's matrix **W<sub>l</sub>** of shape **nxm** and a bias' matrix **B<sub>l</sub>** of shape **1xn**.

###### 2. Forward propagation

1. Propagate batch of samples and store outputs of each layer before (**Z<sub>l</sub>**) and after (**Y<sub>l</sub>**) activation function. 
2. How?
 
 **n** = number of neurons in layer l / **m** = number of neurons in layer l+1 / **k** = batch size
  
First layer:
* Weights (initial) --> W<sub>0</sub> = I<sub>n</sub> (Identity matrix)
* Inputs --> X<sub>l</sub> = X<sub>kxn</sub>
* Biases --> B<sub>l</sub> = 0<sub>1xn</sub>
* Outputs/Activations --> Z<sub>l</sub> = Y<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>0</sub>) + B<sub>l</sub>
* Weights --> W<sub>l</sub> = W<sub>nxm</sub>

For every other layer:
* Inputs --> X<sub>l</sub> = Y<sub>l-1</sub> 
* Biases --> B<sub>l</sub> = B<sub>1xn</sub> 
* Output --> Z<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>l-1</sub>) + B<sub>l</sub>
* Activations --> Y<sub>l</sub> = ActivationFunction(Z<sub>l</sub>)
* Weights --> W<sub>l</sub> = W<sub>nxm</sub> 

Note: the weights of a layer(31) 3047-3612 l refer to the connections between layer l and layer l+1, therefore, the output layer doesn't have a weight's matrix.

###### 3. Backpropagation

1. Calculate the gradients for every weight and every sample.
* For a weight w<sub>ij</sub>, the gradient matrix G<sub>ij</sub> has **k** (batch size) elements.
* Gradient ij = partial derivative of cost function (E) w.r.t. w<sub>ij</sub>
* How?
**i** = neuron in layer l / **j** = neuron in layer l+1
 **n** = number of neurons in layer l / **m** = number of neurons in layer l+1 / **k** = batch size
 
1.1. Chain rule:
![This is an image](https://github.com/AmandaJMendes/MLP/blob/main/tempFileForShare_20220316-144807.jpg)

1.2. Calculate deltas for every layer l(D<sub>l</sub> = D<sub>kxn</sub>)
Last layer:
![This is an image](https://github.com/AmandaJMendes/MLP/blob/main/tempFileForShare_20220316-145401.jpg)

Hidden layers:
![This is an image](https://github.com/AmandaJMendes/MLP/blob/main/gh.png)

2. Compute the average of the k gradients for each weight.
3. Use the gradients in the optimizer function.



