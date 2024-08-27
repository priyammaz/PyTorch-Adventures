class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):

        for param in self.parameters:
            if param.requires_grad:
                param.data -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = 0