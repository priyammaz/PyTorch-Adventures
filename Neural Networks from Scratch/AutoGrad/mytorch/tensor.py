import numpy as np

class Tensor:
    def __init__(self, 
                 data, 
                 requires_grad=False, 
                 grad_fn=None,
                 grad_fn_name=None):
        
        ### Store Passed in Variables ###
        self.data = self._toarray(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad_fn_name = grad_fn_name
        self.shape = self.data.shape

        ### Container to Store Children (output) of Every Operation ###
        ### (i.e. if we add tensor A and B to create C, C is the child of A and child of B) ###
        self.children = []
        
        ### If we actually want to store gradient, Initialize with Zeros ###
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        if self.grad_fn_name is None:
            return f"{self.data}, requires_grad={self.requires_grad}"
        else:
            return f"{self.data}, grad_fn={self.grad_fn_name}"
        
    def _broadcasted_grad_accumulate(self, x, x_grad):
        
        ### This function is crucial and taken from https://github.com/eduardoleao052/Autograd-from-scratch ###
        ### Much of our convenient operations are broadcasting! For example, we can add a tensor of size (A x B)
        ### to another tensor (A x B x C). What broadcasting does is automatically add our (A x B) tensor to C number of 
        ### the (A x B) tensors found in our larger tensor. 

        # (A x B) + (A x B x C) -> (A x B x C)

        ### How does this actually happen? By repeating! Basically our smaller (A x B) tensor is repeated C times to create a 
        ### C * (A x B) which creates an (A x B x C) tensor, and then added to the second (A x B x C) tensor. 

        ### In Neural Networks, a typical dimension we broadcast over is the Batch Dimension. If we have N samples in our neural network ###
        ### we technically can pass in every sample in N one at a time to our neural network. This is obviously very inefficient though ###
        ### and so we broadcast all the components of the neural network over the batch dimension so we can parallize the computation in fast CUDA code ###

        ### Now the problem we have. We want to compute the gradients to update our network based on a batch of samples. Again, technically we could ###
        ### pass in a single sample at a time and then add up (accumulate) the gradients for each sample. But this is a bit slow! So instead we need to just ###
        ### do it all in batches which will cause some hassles. 

        ### Lets say we are solving y = mx + b
        ### b is a (1, 1) tensor, i.e. really just a single value
        ### w is a (1, 1) tensor, i.e. really just a single value
        ### x is a (N x 1) tensor, so we have N samples in our batch and each sample has one feature
        ### y is a (N x 1) tensor, so each input x has an output y with one feature (the thing we are predicting)

        ### Ater we compute y we typically compute our loss, lets say mean squared error (sum(y - y_hat)**2)/N
        ### To learn our coefficient and intercept we need to compute dL/dW and dL/db. Because of chain rule this becomes:

        ### dL/dW = dL/dY * dY/dW
        ### dL/db = dL/dY * dY/db

        ### So this term: dL/dY. For which Y are we doing this? In a batch we have N outputs. Well, we are doing it for all of them, 
        ### and summing them together! (as the operation for the mean loss has a sum inside it). So in the same way, we need to create a 
        ### vector of loss values such as [dL/dY_0, dL/dY_1, dL/dY_2, ... dL/dY_N]. And then this (Nx1) gradient tensor of dL/dY
        ### goes onto the next step of backprop. 

        ### To update w, we saw before that the formula is dL/dW = x^T @ dL/dY. This will be a (1 x N) tensor multiplied by a (N x 1) tensor outputing
        ### a single (1,1) tensor that we need to update our gradients. This "sum" operation across the batch is thus built right into a matrix 
        ### multiplication

        ### To update b on the other hand, the formula was dL/db = sum(dL/dY_i). Thus we need to manually do the sum here!
        ### Therefore we need to accumulate any extra dimensions we have! Our gradient vector for dL/dY is (N x 1), but the 
        ### actual tensor for our single bias value is (1,1). Therefore, on the first dimension we have a discrepancy, 
        ### we have a 1 in our tensor, but N in the gradient, so we must add across the dimension and create an accumulated (1,1) gradient 

        x_shape = x.shape
        grad_shape = x_grad.shape

        assert len(x_shape) == len(grad_shape), "Gradient and tensor shapes must be the same length! Only different by broadcasting"

        for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)):
            ### If our tensor dim is 1 but the grad dim is not, accumulate on that dimension ###
            if (x_dim == 1) and (grad_dim != 1):
                x_grad = x_grad.sum(axis=idx, keepdims=True)
            ### Otherwise verify that our x_dim and grad dim are the same!! 
            else:
                assert (x_dim == grad_dim)

        return x_grad
        
    def backward(self, input_grad=None, child=None):

        if self.requires_grad:

            ### Base Case (dL/dL = 1) for the start of chain rule ###
            if input_grad is None:
                input_grad = np.ones_like(self.data)

            ### Accumulate Gradients ###
            self.grad += input_grad

            ### We are exhausting this backprop path from "child", so we can pop child out ###
            if child is not None:
                self.children.remove(child)
            
            ### If we have a grad function, we can do backward pass ###
            if self.grad_fn is not None:
                ### Until we exhast all the children we cannot move backwards ###
                ### If a single tensor has multiple children, we must backward all the children ###
                ### and accumulate gradients for tensor before backwarding again ###
                if len(self.children) == 0:
                    self.grad_fn(self.grad, self)
    
    def __add__(self, val):

        """
        Sum of two tensors (with accumulation for brodcasting)
        O = A + B
        dO/dA = 1
        dO/dB = 1
        """
    
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)
            
        ### Use Numpy __add__ to actually add tensors together ###
        output = self.data + val.data
        
        ### Define Backward Function ###
        def _add_backward(input_grad, child):

            if self.requires_grad:
                self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self, self_grad)
                self.backward(self_grad, child)

            if val.requires_grad:
                val_grad = input_grad
                val_grad = self._broadcasted_grad_accumulate(val, val_grad)
                val.backward(val_grad, child)

        ### Wrap our output in our tensor object ###
        ### We will compute grad on this if self or val are also tracking gradients ###
        ### We also store the backward_fn for the backward pass ###
        output = Tensor(output,
                        requires_grad=self.requires_grad or val.requires_grad,
                        grad_fn=_add_backward,
                        grad_fn_name="<AddBackward>")
        
        ### This output is the child of the inputs a and b ###
        self._add_child(output)
        val._add_child(output)
        
        return output
    
    def __radd__(self, val):

        """
        add is not an ordered operation, A + B is the same as B + A

        In A + B, our self is A and val is B
        When we do A + B, what is really happening is A.__add__(B). 

        But if A is an integer and B is a Tensor, python integers dont know how to work with our
        own tensor operations. This will throw an error and then try __radd__.  
    
        __radd__ will reverse the operands and do B.__add__(A), using our own Tensor __add__ written above instead.  
        Our __add__ we wrote for the tensor does know how to interface python numbers and tensors so we can then do the operation!

        """
        return self + val
    
    def __sub__(self, val):

        """
        Same as __add__ but now subtraction (with accumulation for broadcasting)
        O = A - B
        dO/dA = 1
        dO/dB = -1
        """
    
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)
            
        ### Use Numpy __add__ to actually add tensors together ###
        output = self.data - val.data
        
        ### Define Backward Function ###
        def _sub_backward(input_grad, child):
            if self.requires_grad:
                self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self, self_grad)
                self.backward(self_grad, child)
                
            if val.requires_grad:
                val_grad = -input_grad
                val_grad = self._broadcasted_grad_accumulate(val, val_grad)
                val.backward(val_grad, child)

        ### Wrap our output in our tensor object ###
        ### We will compute grad on this if self or val are also tracking gradients ###
        ### We also store the backward_fn for the backward pass ###
        output = Tensor(output,
                        requires_grad=self.requires_grad or val.requires_grad,
                        grad_fn=_sub_backward,
                        grad_fn_name="<SubBackward>")
        
        ### This output is the child of the inputs a and b ###
        self._add_child(output)
        val._add_child(output)
        
        return output
    
    def __rsub__(self, val):

        """
        Subtraction is an ordered operation. Lets say we want A - B where A is self and B is val
        if A is not a tensor (i.e. an int or float), __sub__ will throw an error as it doesnt know
        how to do an operation with our own tensor.

        This will enter __rsub__ where we flip the operands where B is now self and A is val. If we want
        A - B, we need to do -1 * B + A, using our __add__. 

        There are a bunch of ways to handle these exceptions, this is just one of them!
        """

        return -1 * self + val

    def __mul__(self, val):

        """
        Element-wise multiplication of two tensors (with accumulation for broadcasting)

        O = A * B
        dO/dA = B
        do/dB = A
        """

        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)
            
        output = self.data * val.data

        def _mul_backward(input_grad, child):

            if self.requires_grad:
                self_grad = input_grad * val.data
                self_grad = self._broadcasted_grad_accumulate(self, self_grad)
                self.backward(self_grad, child)
            
            if val.requires_grad:
                val_grad = input_grad * self.data
                val_grad = self._broadcasted_grad_accumulate(val, val_grad)
                val.backward(val_grad, child)

        output = Tensor(output, 
                        requires_grad=self.requires_grad or val.requires_grad, 
                        grad_fn=_mul_backward,
                        grad_fn_name="<MulBackward>")
        
        self._add_child(output)
        val._add_child(output)

        return output
    
    def __rmul__(self, val):
        return self * val

    def __matmul__(self, val):

        """
        Expanding __mul__ to matrix multiplication (@)

        O = A @ B

        """
        
        if not isinstance(val, Tensor):
            val = Tensor(val)
        
        ### Use numpy Matmul Operation ###
        output = self.data @ val.data

        ### Define Matmul Backward ###
        def _matmul_backward(input_grad, child):
            if self.requires_grad:
                self_grad = input_grad @ val.data.swapaxes(-1,-2)
                self.backward(self_grad, child)

            if val.requires_grad:
                val_grad = self.data.swapaxes(-1,-2) @ input_grad
                val.backward(val_grad, child)

        ### Convert to Tensor ###
        output = Tensor(output,
                        requires_grad=self.requires_grad or val.requires_grad,
                        grad_fn=_matmul_backward,
                        grad_fn_name="<MatmulBackward>")

        ### This output is the child of the inputs a and b ###
        self._add_child(output)
        val._add_child(output)

        return output
    
    def __truediv__(self, val):

        """
        Element-wise Division of two tensors (accumulated grad for broadcasting)

        O = A/B
        dO/dA = 1/B
        dO/dB = -A/B^2

        """

        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)

        output = self.data / val.data

        def _div_backward(input_grad, child):
            if self.requires_grad:
                self_grad = input_grad / val.data
                self_grad = self._broadcasted_grad_accumulate(self, self_grad)
                self.backward(self_grad, child)

            if val.requires_grad:
                val_grad = input_grad * -1 * self.data / (val.data**2)
                val_grad = self._broadcasted_grad_accumulate(val, val_grad)
                val.backward(val_grad, child)
        
        ### Convert to Tensor ###
        output = Tensor(output,
                        requires_grad=self.requires_grad or val.requires_grad,
                        grad_fn=_div_backward,
                        grad_fn_name="<DivBackward>")

        ### This output is the child of the inputs a and b ###
        self._add_child(output)
        val._add_child(output)

        return output

    def __rtruediv__(self, val):
        
        """
        Div is an ordered operation. Lets say we want A/B, in the case of __div__ A is self and B is val. 
        if A is not a Tensor (i.e. an int or float), A / B will throw an error beacuse we only can divide a tensor by a tensor
        In this case, __rtruediv__ will be called where A is now val and B is self (the operands have been flipped)
        We can then convert A (our non-tensor) which is in val to a tensor and then perform val / self to call __div__ again where
        A and B are both now tensors
        """
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)

        return val / self

    def __pow__(self, exponent):

        """
        Element-wise exponentiation of matrix (assuming exponent is non-learnable)
        O = A^K
        dO/dA = K * A^(k-1)
        """

        output = self.data ** exponent

        def _pow_backward(input_grad, child):
            self_grad = input_grad * (exponent * self.data ** (exponent-1))
            self.backward(self_grad, child)

        output = Tensor(output,
                        requires_grad=self.requires_grad,
                        grad_fn=_pow_backward,
                        grad_fn_name="<PowBackward>")
        
        self._add_child(output)

        return output
    
    def exp(self):

        """
        Element-wise exponentiation of the base e
        O = e^A
        dO/dA = e^A
        """

        output = np.exp(self.data)

        def _exp_backward(input_grad, child):   
            self_grad = input_grad * np.exp(self.data)
            self.backward(self_grad, child)
        
        output = Tensor(output, 
                        requires_grad=self.requires_grad,
                        grad_fn=_exp_backward, 
                        grad_fn_name="<ExpBackward>")
        
        self._add_child(output)

        return output
    
    def log(self):

        """
        Element-wise log with base e
        O = log(A)
        dO/dA = 1/a
        """

        output = np.log(self.data)

        def _log_backward(input_grad, child):   
            self_grad = input_grad * (1/self.data)
            self.backward(self_grad, child)
        
        output = Tensor(output, 
                        requires_grad=self.requires_grad,
                        grad_fn=_log_backward, 
                        grad_fn_name="<LogBackward>")
        
        self._add_child(output)

        return output

    def sum(self, dim=-1, keepdims=False):

        """
        Sum across a dimension!

        O = sum([a_1, a_2, a_3, ...])

        Remember, sum operations just channel the incoming gradients from the later computational paths. 
        This means our input_gradient coming from operations after the sum here just needs to be copied to all 
        values of [a_1, a_2, a_3, ...]

        dO/da_1 = input_grad
        dO/da_2 = input_grad
        dO/da_3 = input_grad
        ...

        """

        output = self.data.sum(axis=dim, keepdims=keepdims)

        def _sum_backward(input_grad, child):

            ### Add dimensions to input grad to match self
            grad_dims = len(input_grad.shape)
            self_dims = len(self.shape)

            if grad_dims != self_dims:
                diff = self_dims - grad_dims    
                for _ in range(diff):
                    input_grad = np.expand_dims(input_grad, axis=-1)
            
            self_grad = input_grad * np.ones((self.shape))
    
            self.backward(self_grad, child)
        
        output = Tensor(output,
                        requires_grad=self.requires_grad,
                        grad_fn=_sum_backward,
                        grad_fn_name="<SumBackward>")
        
        self._add_child(output)

        return output
    
    def mean(self, dim=-1):

        """
        Almost identical to Sum across a dimension, except divided by the constant of the number of elements summed

        O = sum([a_1, a_2, a_3, ..., a_N]) / N

        Remember, sum operations just channel the incoming gradients from the later computational paths. 
        This means our input_gradient coming from operations after the sum here just needs to be copied to all 
        values of [a_1, a_2, a_3, ...]

        dO/da_1 = input_grad/N
        dO/da_2 = input_grad/N
        dO/da_3 = input_grad/N
        ...

        """
        
        output = self.data.mean(axis=dim)

        def _mean_backward(input_grad, child, dim=dim):
                
                ### Add dimensions to input grad to match self
                grad_dims = len(input_grad.shape)
                self_dims = len(self.shape)

                if grad_dims != self_dims:
                    diff = self_dims - grad_dims    
                    for _ in range(diff):
                        input_grad = np.expand_dims(input_grad, axis=-1)


                ### We average over the dim dimension ###
                ### and averaging is just a sum / constant ###
                ### where the constant is the num elements in that dim ###
                ### So we can multiply our input_grad by the 1/constant ###

                if isinstance(dim, int):
                    dim = [dim]
                elif isinstance(dim, tuple):
                    dim = list(dim)
                
                dim_sizes = [self.shape[i] for i in dim] if dim != -1 else list(self.shape)
                num_vals_avged = np.prod(dim_sizes)

                self_grad = input_grad * np.ones((self.shape)) / num_vals_avged
                self.backward(self_grad, child)

        output = Tensor(output,
                        requires_grad=self.requires_grad,
                        grad_fn=_mean_backward,
                        grad_fn_name="<MeanBackward>")
        
        self._add_child(output)

        return output
        
    def reshape(self, shape):

        """
        If we reshape our tensor, we just need to reshape the incoming identically! 
        Remember, gradients of a tensor are the same shape as the tensor itself, so we 
        just need to make sure that our gradient index coorespond to the correct tensor index. 
        """
        
        output = self.data.reshape(shape)

        def _reshape_backward(input_grad, child, shape=shape):

            if self.requires_grad:
                self_grad = input_grad.reshape(shape)
                self.backward(self_grad, child)

        output = Tensor(output, 
                        requires_grad=self.requires_grad, 
                        grad_fn=_reshape_backward, 
                        grad_fn_name="ReshapeBackward")
        
        self._add_child(output)
        
        return output

    def _toarray(self, input):

        """
        Helper to convert an input to a numpy array
        """
        if isinstance(input, np.ndarray):
            return input
        elif isinstance(input, Tensor):
            return input.data
        else:
            return np.array(input)

    def _add_child(self, child_tensor):

        """
        Helper function to add a tensor as a child of an operation
        """
        
        if not isinstance(child_tensor, Tensor):
            raise Exception("Children of Tensors must also be a Tensor")
        self.children.append(child_tensor)