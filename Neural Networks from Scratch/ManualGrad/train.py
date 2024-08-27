import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradTensor:

    def __init__(self, params):
        self.params = params
        self.grad = None
    
    def _zero_grad(self):
        self.grad = None
    
class Linear:

    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w = GradTensor(np.random.normal(scale=0.02, size=(in_features, out_features)))
        if bias:
            self.b = GradTensor(np.random.normal(size=(1,out_features)))
        else:
            self.b = None
    
    def forward(self, x):
        
        # For backprop, dL/dW will need X^t, so save X for future use 
        self.x = x

        # x has shape (B x in_features), w has shape (in_features x out_features)
        proj = x @ self.w.params
        if self.bias:
            # proj has shape (B x out_features) and bias has shape (1 x out_features)
            proj += self.b.params
        
        return proj 
    
    def backward(self, output_grad):
        
        ### Parameter Gradients ###

        # Derivative wrt W: X^t @ output_grad
        self.w.grad = self.x.T @ output_grad

        # Derivate of Bias is just the output grad summed along batch 
        self.b.grad = output_grad.sum(axis=0, keepdims=True)

        # We need derivative wrt X for next step
        input_grad = output_grad @ self.w.params.T

        return input_grad

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true-y_pred)**2)
    
    def backward(self):
        n = self.y_pred.shape[0]
        grad = -(2/n)*(self.y_true-self.y_pred)
        return grad
    
class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, output_grad):
        sigmoid_x = self.forward(self.x)
        sigmoid_grad = sigmoid_x * (1 - sigmoid_x)
        input_grad = sigmoid_grad * output_grad
        return input_grad

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):

        for layer in self.layers:
            input = layer.forward(input)

        return input
        
    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                parameters.append(layer.w)
                if layer.bias:
                    parameters.append(layer.b)

        return parameters

class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            assert param.grad.shape == param.params.shape, "Something wrong"
            param.params -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class LinearScheduler:
    def __init__(self, optimizer, total_training_steps, warmup_steps):
        assert warmup_steps <= total_training_steps

        self.optimizer = optimizer
        max_lr = self.optimizer.lr

        warmup = list(np.linspace(0,max_lr, warmup_steps))
        main = list(np.linspace(max_lr, 0, total_training_steps-warmup_steps))

        self.scheduler = warmup + main
        self.step_counter = 0

        self.optimizer.lr = self.scheduler[self.step_counter]
    
    def step(self):
        self.optimizer.lr = self.scheduler[self.step_counter]
        self.step_counter += 1
    

if __name__ == "__main__":
    x_train = np.arange(-10,10,0.1).reshape(-1,1)
    y_train = 3*x_train**2 + 2 + np.random.normal(scale=5, size=(x_train.shape[0], x_train.shape[1]))

    network = NeuralNetwork()
    network.add(Linear(1,32))
    network.add(Sigmoid())
    network.add(Linear(32,64))
    network.add(Sigmoid())
    network.add(Linear(64,64))
    network.add(Sigmoid())
    network.add(Linear(64,32))
    network.add(Sigmoid())
    network.add(Linear(32,1))

    loss_func = MSELoss()
    optimizer = SGD(network.parameters(), 1e-3)
    scheduler = LinearScheduler(optimizer, 25000, 2500)
    
    losses = []
    for idx in tqdm(range(25000)):
        output = network(x_train)

        loss = loss_func.forward(output, y_train)
        loss_grad = loss_func.backward()
        network.backward(loss_grad)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss)

        if idx % 250 == 0:
            print("Loss:", loss)
    pred = network(x_train)

    plt.plot(y_train, label="truth")
    plt.plot(pred, label="pred")
    plt.legend()
    plt.show()

    plt.plot(losses)
    plt.show()