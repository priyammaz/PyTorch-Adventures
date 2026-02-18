import cupy as np ### USING NP NOTATION BUT CUPY ALMOST IDENTICAL TO NUMPY

class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            assert param.grad.shape == param.params.shape, "Something has gone horribly wrong"
            param.params -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class Adam:

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8):

        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ### Create Momentum Vector for Each Parameter ###
        self.m = [np.zeros_like(p.params) for p in parameters]
        self.v = [np.zeros_like(p.params) for p in parameters]

        ### Step Index for Bias Correction ###
        self.t = 0

    def step(self):

        self.t += 1

        for i, param in enumerate(self.parameters):

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            bias_corrected_m = self.m[i] / (1 - self.beta1**self.t)
            bias_corrected_v = self.v[i] / (1 - self.beta2**self.t)
            
            param.params -= self.lr * bias_corrected_m / (np.sqrt(bias_corrected_v) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

