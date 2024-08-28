# ManualGrad

As opposed to AutoGrad that you find in PyTorch, we will be starting our Neural Network journey with ManualGrad! Lets quickly recap Neural Networks again! There are typically two things that make a Nerual Network unique:

- Linear Projection
- Non-Linear Activation Function

### Linear Projection 

This entire operation is typically performed through a matrix mulitiplication and sum. We multiply our input by some weights and then add a constant as a bias. 

### Non-Linear Activation

If we stack a bunch of linear projections on top of each other, the model can only ever learn a linear relationship. This is why non-linear functions are super important! By sandwiching nonlinearity between our linear operations, we can then learn a non-linear relationships! There are a ton of activation functions we can pick from, but today we will go with the classic Sigmoid function. 

### Derivation of the Derivatives

As I stated before, there are only two operations that we will be looking at:

- Linear Projection: Y = X @ W + b
  - *W* and *b* are the weights and biases that are learnable
- Sigmoid Activation Function 
  - $\sigma(x) = \frac{1}{1 + \exp(-x)}$

What we need to derive is the derivative of all of these terms! The complete derivation can be found in my [derivations](Backpropagation.pdf), but I will just give the main results here!

If we have $\hat{y} = XW + b$ then we need to compute for our derivatives of our parameters with respect to some loss function $L$ so we can use them for gradient descent:
$$\frac{dL}{dW} = \frac{dL}{d \hat{y}}\frac{d \hat{y}}{dW} = X^T\frac{dL}{d \hat{y}}$$
$$\frac{dL}{db} = \frac{dL}{d \hat{y}}\frac{d \hat{y}}{db} = \Sigma_{n=1}^N\frac{dL}{d \hat{y_i}}$$

For backpropagation to the next layer we will also need $\frac{dL}{dX}$ so we can compute that as:
$$\frac{dL}{dX} = \frac{dL}{d \hat{y}}W^T$$ 

For our activation function we will again be using Sigmoid, which also has its own easy derivative! 

$$\frac{d \sigma(x)}{d x} = \sigma(x) (1 - \sigma(x)) $$
