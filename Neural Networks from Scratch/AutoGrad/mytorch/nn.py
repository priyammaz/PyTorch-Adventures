import math
import numpy as np
from .tensor import Tensor

######################
### Generic Module ###
######################
class Module:

    def parameters(self):
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor):
                params.append(val)

            if isinstance(val, Linear):
                linear_params = val.parameters()
                params.extend(linear_params)

        return params

    def __repr__(self):
        model_name = self.__class__.__name__
        model_string = f"{model_name}(\n"
        for key, val in self.__dict__.items():
            model_string += f"  ({key}): {val}\n"
        model_string += ")"
        return model_string
    
    def __call__(self, x):
        return self.forward(x)

####################
### Linear Layer ###
####################
class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):

        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features

        ### Initialize Weights as Described in nn.Linear ###
        ### https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        k = math.sqrt(1/in_features)
        self.W = Tensor(np.random.uniform(low=-k, high=k, size=(in_features, out_features)), requires_grad=True)

        if self.bias:
            self.b = Tensor(np.random.uniform(low=-k, high=k, size=(1, out_features)), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
    
    def forward(self, x):
        output = x @ self.W

        if self.bias:
            output = output + self.b

        return output

############################
### Activation Functions ###
############################
class Sigmoid:
    def __init__(self):
        pass

    def __repr__(self):
        return "Sigmoid()"
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return 1 / (1 + (-1*x).exp())

###################### 
### Loss Functions ###
######################
class CrossEntropyLoss:

    def __init__(self):
        pass

    def __call__(self, pred, labels):
        return self.forward(pred, labels)

    def forward(self, pred, labels):
         
        ### Softmax Predicted Logits ###
        softmax_pred = pred.exp() / pred.exp().sum(dim=-1, keepdims=True)

        ### Compute Log Probs ###
        log_probs = softmax_pred.log()

        ### Multiply by OHE Labels ###
        prod = log_probs * labels

        ### Sum across each sample ###
        loss_per_sample = -1 * prod.sum(dim=-1)
        
        ### Average Loss across batch ###
        loss = loss_per_sample.mean()

        return loss
    
class MSELoss:

    def __init__(self):
        pass

    def __call__(self, pred, labels):
        return self.forward(pred, labels)
    
    def forward(self, pred, labels):
        return ((pred-labels)**2).mean(dim=0)