# MLP

#### Codes
* mlp.py --> Main code
* metrics.py --> Implementation of some metrics for classification and regression problems
* optimizers.py --> Implementation of SGD function
* classification.py --> Usage of the MLP for classification
* regression.py --> Usage of the MLP for regression

### Theory
* derivativesActs.pdf --> Derivatives deduction for the activation functions
* derivativesMetrics.pdf --> Derivatives deduction for the metrics functions
* readme0.jpg; readme1.jpg; readme2.png --> Mentioned throughout backpropagation section of the README file 

###### 1. Initialize MLP

1. Set initial weights and bias for each layer.
2. A layer **l** with **n** neurons followed by a layer **l+1** with **m** neurons has a weight matrix **W<sub>l</sub>** of shape **nxm** and a bias matrix **B<sub>l</sub>** of shape **1xn**.

###### 2. Forward propagation

1. Propagate batch of samples and store outputs of each layer before (**Z<sub>l</sub>**) and after (**Y<sub>l</sub>**) activation function. 
2. How?

**n** = number of neurons in layer l / **m** = number of neurons in layer l+1 / **k** = batch size
  
First layer:
* Weights (initial) --> W<sub>0</sub> = I<sub>n</sub> (Identity matrix)
* Inputs --> X<sub>l</sub> = X<sub>kxn</sub>
* Bias --> B<sub>l</sub> = 0<sub>1xn</sub>
* Outputs/Activations --> Z<sub>l</sub> = Y<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>0</sub>) + B<sub>l</sub>
* Weights --> W<sub>l</sub> = W<sub>nxm</sub>

For every other layer:
* Inputs --> X<sub>l</sub> = Y<sub>l-1</sub> 
* Bias --> B<sub>l</sub> = B<sub>1xn</sub> 
* Output --> Z<sub>l</sub> = Dot product(X<sub>l</sub>, W<sub>l-1</sub>) + B<sub>l</sub>
* Activations --> Y<sub>l</sub> = ActivationFunction(Z<sub>l</sub>)
* Weights --> W<sub>l</sub> = W<sub>nxm</sub> 

Note: the weights of a layer l refer to the connections between layer l and layer l+1, therefore, the output layer doesn't have a weight matrix.

###### 3. Backpropagation

1. Calculate the gradients for every weight and every sample.
* For a weight w<sub>ij</sub>, the gradient matrix G<sub>ij</sub> has **k** (batch size) elements.
* Gradient ij = partial derivative of cost function (E) w.r.t. w<sub>ij</sub>
* How?

  **i** = neuron in layer l / **j** = neuron in layer l+1 / **n** = number of neurons in layer l / **m** = number of neurons in layer l+1 / **k** = batch size
 
  1.1. Chain rule (readme0.jpg).

  1.2. Calculate deltas for every layer l (D<sub>l</sub> = D<sub>kxn</sub>).

  * Last layer (readme1.jpg).
  * Hidden layers (readme2.png).
  
  1.3. Calculate the gradients for every weight using the chain rule mentioned in 1.1. For the bias, the gradients are the deltas themselves. 

2. Compute the average of the k gradients for each weight and bias.
3. Use the gradients in the optimizer function.



