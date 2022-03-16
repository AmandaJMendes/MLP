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
* How?(31) 3047-3612
**i** = neuron in layer l / **j** = neuron in layer l+1
 **n** = number of neurons in layer l / **m** = number of neurons in layer l+1 / **k** = batch size
 
1.1. Chain rule:
![This is an image](https://lh3.googleusercontent.com/9_HpCwDk5eY1kxPhVoCCf3lu4-Fargr3VADuV5l0awMi8N1BSXlI106xbidvK_p-oAtBYYLIavPm7CTSs-Qz_sHS2X2BBcCNoOyU-BbxKeqbh_VkhTw59tNWpOJtm0Hr1roU3o9EI8mMYDLMTeMvtOntDChNXJCBGgDhWfRr5F_GsLECfnT-7M5W9F3epCKggVkCD3xLbZ46uk2eTDBIv-KziuWoAdzYzgtSP1SPmuqktBHxoQ3tcZr4EXLt70QCr8uX_jXr7LWCQ8N7NX_ZcrFxQxNYiS1dA9SFKJRnAxojddea8yVdntu4iBoA0KGvK9FTF-3V72VSI5bf_JT9SlfAU2IW9VRqpDhyp921Fx706ATPP9xo2UwK9qC5jYIVacopNB-4rGzy-55FI0Q41xm9iRBJTZPeEQvpZlk1hxlD9UKTM-ZPLnWe94nW-CbdDhcY9V2GOVvmgRud3w4zwt55jlqsHV84and6zJYO7oj3rp5Jaj-WxZKVcoCEnan6ekDtJRUkFblTBqpk0h3jekTQTWQd7ffFRldK4KgtDcVKZxyms73IXaCMZezYBaS0w0LENW9xIn-nW4Q38gusIYVYBv2SkAaDyX9FO_cbmHptu7yUj6mG_tlj7hOQLa33Vglf4GkDNyQZBM-xnUIeR1FYX4cqVmxeB2DY4k6EF6In9tME_h5lxDB3xEgPl6rNxElIdRA3W3Uox0AfIQh1ivt9gQ=w655-h591-no?authuser=0)
1.2. Calculate deltas for every layer l(D<sub>l</sub> = D<sub>kxn</sub>)
Last layer:



