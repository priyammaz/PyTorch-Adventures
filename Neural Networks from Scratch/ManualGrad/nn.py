import cupy as np
import math

class Operation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class GradLayer(Operation):

    def parameters(self):
        params = []
        for attr_name, attr_values in self.__dict__.items():
            if isinstance(attr_values, GradTensor):
                params.append(attr_values)
            
            elif isinstance(attr_values, GradLayer):
                params.extend(attr_values.parameters())

        return params

class GradTensor:
    def __init__(self, params):
        self.params = params
        self.shape = params.shape
        self.grad = None
    
    def _zero_grad(self):
        self.grad = None

#######################
### STANDARD LAYERS ###
#######################

class Linear(GradLayer):
    """
    Optimized Linear layer for CuPy
    y = x @ W + b
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = GradTensor(
            np.random.normal(
                scale=0.02,
                size=(in_features, out_features)
            )
        )

        if bias:
            self.bias = GradTensor(
                np.zeros((1, out_features))
            )
        else:
            self.bias = None

    def forward(self, x):
        # Ensure contiguous float32 arrays
        self.x = x if x.flags['C_CONTIGUOUS'] and x.dtype == np.float32 else np.ascontiguousarray(x, dtype=np.float32)

        out = np.empty((self.x.shape[0], self.out_features), dtype=np.float32)
        np.matmul(self.x, self.weight.params, out=out)

        if self.bias is not None:
            out += self.bias.params  # broadcasted in-place

        return out

    def backward(self, grad_output):
        grad_output = grad_output if grad_output.flags['C_CONTIGUOUS'] and grad_output.dtype == np.float32 else np.ascontiguousarray(grad_output, dtype=np.float32)

        # Grad w.r.t weight: X^T @ grad_output
        grad_W = np.empty_like(self.weight.params)
        np.matmul(self.x.T, grad_output, out=grad_W)
        self.weight.grad = grad_W

        # Grad w.r.t bias: sum over batch, in-place
        if self.bias is not None:
            self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)

        # Grad w.r.t input: grad_output @ W^T
        grad_input = np.empty_like(self.x)
        np.matmul(grad_output, self.weight.params.T, out=grad_input)

        return grad_input

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
    
class MSELoss(Operation):
    """
    L = E[(y-y_hat)^2]
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred)**2)
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        grad = -(2/batch_size) * (self.y_true - self.y_pred)
        return grad
    
    def __repr__(self):
        return "MSELoss()"

class SLOW_SoftMax(Operation):
    """
    Softmax normalization to convert logits -> probabilities
    Backward explicitly computes the jacobian for each sample
    """
    def forward(self, x):
        self.x = x
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, output_grad):
        gradient = np.zeros_like(self.probs)
        batch_size = len(self.x)
        for i in range(batch_size):
            sample_probs = self.probs[i]
            j = -sample_probs.reshape(-1,1) * sample_probs.reshape(1,-1)
            j[np.diag_indices(j.shape[0])] = sample_probs * (1 - sample_probs)
            a = output_grad[i] @ j
            gradient[i] = a
       
        return gradient

    def __repr__(self):
        return "SLOW_SoftMax()"

class SoftMax(Operation):
    """
    Softmax normalization to convert logits -> probabilities
    Backward vectorizes gradient computation
    """
    def forward(self, x):
        self.x = x
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, output_grad):
        dot_product = np.sum(output_grad * self.probs, axis=-1, keepdims=True)
        gradient = self.probs * (output_grad - dot_product)
        return gradient

    def __repr__(self):
        return "SoftMax()"

class SLOW_CrossEntropyLoss(Operation):
    """
    Cross Entropy Loss that assumes inputs are probabilities (already softmaxed).
    Targets are integer class indices (no one-hot).
    """
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) integer class indices
        y_pred: (batch_size, num_classes) probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        batch_size = y_pred.shape[0]
        # Pick probability of the correct class for each sample
        correct_class_probs = y_pred[np.arange(batch_size), y_true]
        # Compute loss
        loss = -np.mean(np.log(correct_class_probs + 1e-12))  # epsilon for safety
        return loss

    def backward(self):
        """
        Gradient wrt probabilities
        dL/dy_pred = -y_true / y_pred
        But since y_true is an index, we only subtract at the correct class.
        """
        batch_size, num_classes = self.y_pred.shape
        grad = np.zeros_like(self.y_pred)
        # Only the correct class contributes to loss: -y_i / p_c
        grad[np.arange(batch_size), self.y_true] = -1 / (self.y_pred[np.arange(batch_size), self.y_true] + 1e-12)
        return grad
    
    def __repr__(self):
        return "SLOW_CrossEntropyLoss()"

class CrossEntropyLoss(Operation):
    def forward(self, y_true, logits):
        """
        y_true: [batch_size] integer class indices
        logits: [batch_size, num_classes] raw scores (NOT softmaxed)
        """
        self.y_true = y_true
        self.logits = logits
        # numerically stable softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # pick the probabilities of the correct class for each sample
        batch_size = y_true.shape[0]
        correct_class_probs = self.probs[np.arange(batch_size), y_true]
        # cross-entropy loss
        loss = -np.mean(np.log(correct_class_probs + 1e-12))
        return loss

    def backward(self):
        """
        Gradient of loss wrt logits (softmax + cross entropy simplified)
        """
        batch_size, num_classes = self.logits.shape
        grad = self.probs.copy()
        # subtract 1 at the correct class index
        grad[np.arange(batch_size), self.y_true] -= 1
        return grad
    
    def __repr__(self):
        return "CrossEntropyLoss()"

class Sigmoid(Operation):
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, output_grad):
        sigmoid_x = self.forward(self.x)
        sigmoid_grad = sigmoid_x * (1 - sigmoid_x)
        input_grad = sigmoid_grad * output_grad
        return input_grad

    def __repr__(self):
        return "Sigmoid()"

class ReLU(Operation):
    def forward(self, x):
        self.x = x
        return np.clip(x, a_min=0, a_max=None)
    
    def backward(self, output_grad):
        grad = np.zeros_like(self.x)
        grad[self.x > 0] = 1
        return grad * output_grad
    
    def __repr__(self):
        return "ReLU()"
    
class GELU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # The approximation is used for numerical stability and efficiency
        y = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return y
    
    def backward(self, output_grad):
        # Derivative of the approximation
        tanh_val = np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))
        sech_sq_val = 1 - np.power(tanh_val, 2)
        
        # This is a part of the derivative of the approximation
        approx_deriv_part = 0.5 * (1 + tanh_val) + self.x * 0.5 * sech_sq_val * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(self.x, 2))
        
        return output_grad * approx_deriv_part

    def __repr__(self):
        return "GELU()"

#######################################
### LAYERS FOR CONVOLUTIONAL MODELS ###
#######################################

class SLOW_Conv2d(GradLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        ### Xavier Initialization
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        self.weight = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        batch_size, c, h, w = x.shape

        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            self.x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), "constant")
        else:
            self.x_padded = x

        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride + self.kernel_size
                        w_start, w_end = j*self.stride, j*self.stride + self.kernel_size
                        
                        ### Computed 1 Output Channel At a Time ###
                        out[b, oc, i, j] = np.sum(
                            self.x_padded[b, :, h_start:h_end, w_start:w_end] * self.weight[oc]
                        ) + self.bias[oc]

        return out
    
    def backward(self, grad_out):
        batch_size, _, h_padded, w_padded = self.x_padded.shape
        grad_x_padded = np.zeros_like(self.x_padded)
        grad_w = np.zeros_like(self.weight)
        grad_b = np.zeros_like(self.bias)

        out_h, out_w = grad_out.shape[2], grad_out.shape[3]

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride + self.kernel_size
                        w_start, w_end = j*self.stride, j*self.stride + self.kernel_size

                        grad_w[oc] += grad_out[b, oc, i, j] * self.x_padded[b, :, h_start:h_end, w_start:w_end]     
                        grad_x_padded[b, :, h_start:h_end, w_start:w_end] += grad_out[b, oc, i, j] * self.weight[oc]
                
                grad_b[oc] += np.sum(grad_out[b,oc])
        
        if self.padding > 0:
            grad_x = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_x = grad_x_padded

        self.grad_w = grad_w
        self.grad_b = grad_b

        return grad_x

    def __repr__(self):
        return (f"SLOW_Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={self.bias})")
    
class Conv2d(GradLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier Initialization
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        weights = np.random.uniform(-limit, limit, 
                                    (out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            bias = np.zeros((1, out_channels)) 

        # Create Linear Layer #
        self.conv_linear = Linear(in_channels * kernel_size * kernel_size, out_channels, bias=bias is not None)

        # Set Weights of Linear Layer with our Xavier Init ###
        self.conv_linear.weight.params = weights.reshape(in_channels * kernel_size * kernel_size, out_channels)
        if bias:
            self.conv_linear.bias.params = bias

    def im2col(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if P > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        else:
            x_padded = x

        out_h = (H + 2*P - K)//S + 1
        out_w = (W + 2*P - K)//S + 1
        n_patches = out_h * out_w

        ### Extract image patches for mat mul (convolution) ###
        cols = np.zeros((B, n_patches, C*K*K))
        for i in range(out_h):
            for j in range(out_w):
                patch = x_padded[:, :, i*S:i*S+K, j*S:j*S+K]
                cols[:, i*out_w + j, :] = patch.reshape(B, -1)

        return cols, out_h, out_w

    def col2im(self, cols, x_shape, out_h, out_w):
        B, C, H, W = x_shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if P > 0:
            x_padded = np.zeros((B, C, H + 2*P, W + 2*P))
        else:
            x_padded = np.zeros((B, C, H, W))

        for i in range(out_h):
            for j in range(out_w):
                ### Extract corresponding column that goes with this patch in the ####
                ### original image. This will be used in our backward pass on our gradients ###
                ### to return our grad tensor from a simple matrix (for matmul in linear) ###
                ### to the normal shape of a convolution. Due to the overlap between patches 
                ### we want to accumulate up all the overlapping contributions to gradients ###
                ### Hence we use += ###
                patch = cols[:, i*out_w + j, :].reshape(B, C, K, K)
                x_padded[:, :, i*S:i*S+K, j*S:j*S+K] += patch

        if P > 0:
            return x_padded[:, :, P:-P, P:-P]
        
        return x_padded

    def forward(self, x):
        self.x_shape = x.shape

        ### Convert to Matmul with Im2Col ###
        self.x_cols, out_h, out_w = self.im2col(x)
        B, n_patches, C_K_K = self.x_cols.shape

        ### Flatten for Matmul in Linear Layer ###
        x_cols_flat = self.x_cols.reshape(B * n_patches, C_K_K)

        ### Pass through Linear Layer ###
        out_cols_flat = self.conv_linear.forward(x_cols_flat)

        # Reshape back to (B, out_channels, out_h, out_w)
        out_cols = out_cols_flat.reshape(B, n_patches, self.out_channels)
        out = out_cols.transpose(0, 2, 1).reshape(B, self.out_channels, out_h, out_w)

        return out

    def backward(self, grad_out):
        B, C, H, W = self.x_shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        ### Compute Output Shape ###
        out_h = (H + 2*P - K)//S + 1
        out_w = (W + 2*P - K)//S + 1

        ### Reshape grad_out to (B*n_patches, out_channels) so its in the form that ###
        ### nn.Linear expects our gradients to be in ###
        n_patches = out_h * out_w
        grad_cols = grad_out.reshape(B, self.out_channels, -1).transpose(0, 2, 1)
        grad_cols_flat = grad_cols.reshape(B * n_patches, self.out_channels)

        ### Compute Gradients with our Linear Layer ###
        grad_x_cols_flat = self.conv_linear.backward(grad_cols_flat)

        ### Save the gradients in this layer reshaping to our original weights shape ####
        self.grad_w = self.conv_linear.weight.grad.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.grad_b = self.conv_linear.bias.grad

        ### Reshape back to (B, n_patches, C*K*K) in the form the convolution expects it ###
        grad_x_cols = grad_x_cols_flat.reshape(B, n_patches, self.in_channels * K * K)

        ### Convert back to original image shape ###
        grad_x = self.col2im(grad_x_cols, self.x_shape, out_h, out_w)

        return grad_x

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={True if self.conv_linear.bias.params is not None else False})")

class Dropout(Operation):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # By default, layers are in training mode

    def forward(self, x):
        if self.training:
            # Create mask of zeros and ones
            self.mask = (np.random.rand(*x.shape) >= self.p) 
            ### Scaling so mask divides all non-masked values by 1/p to maintain
            ### the overall variance of the tensor 
            self.mask = self.mask / (1.0 - self.p)
            return x * self.mask
        else:
            return x  # No dropout in evaluation

    def backward(self, output_grad):
        if self.training:
            return output_grad * self.mask
        else:
            return output_grad
        
    def __repr__(self):
        return f"Dropout(p={self.p})"

class BatchNorm2d(GradLayer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        ### Learnable Parameters that allow for scale and shift of the activations ###
        self.gamma = GradTensor(np.ones((1, num_features, 1, 1)))
        self.beta = GradTensor(np.zeros((1, num_features, 1, 1)))

        ### Running Buffers of Estimates ###
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.training = True

    def forward(self, x):
        self.x = x

        if self.training:
            ### Average over everything but the channels ###
            mean = x.mean(axis=(0,2,3), keepdims=True)
            var = x.var(axis=(0,2,3), keepdims=True)

            ### Save For Backward Pass ###
            self.mean = mean
            self.var = var

            ### Normalize ###
            self.x_hat = (x - mean) / np.sqrt(var + self.eps)

            ### Update Running States (with momentum) ###
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        ### Scale and Shift with Gamma/Beta ###
        out = self.gamma.params * self.x_hat + self.beta.params

        return out
    
    def backward(self, output_grad):
        N, C, H, W = self.x.shape

        ### Update Gamma and Beta ###
        ### dL/d_gamma = d_L/d_out * d_out/d_gamma 
        self.gamma.grad = np.sum(output_grad * self.x_hat, axis=(0,2,3), keepdims=True)
        self.beta.grad = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)

        ### gradient d_l/d_x ###
        ### d_l/d_x = d_l/d_x_hat * d_x_hat / d_x
        # Gradient w.r.t. input
        dx_hat = output_grad * self.gamma.params
        var_eps = self.var + self.eps

        mean1 = np.mean(dx_hat, axis=(0,2,3), keepdims=True)
        mean2 = np.mean(dx_hat * self.x_hat, axis=(0,2,3), keepdims=True)
        dx = (dx_hat - mean1 - self.x_hat*mean2) / np.sqrt(var_eps)

        return dx
    
    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"

class Flatten(Operation):
    def forward(self, x):
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def __repr__(self):
        return "Flatten()"

###############################
### LAYERS FOR TRANSFORMERS ###
###############################

class Embedding(GradLayer):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weight = GradTensor(
            np.random.normal(
                scale=0.02,
                size=(vocab_size, embed_dim)
            )
        )
    
    def forward(self, x):
        """
        This just indexes our embedding matrix
        """
        self.x = x
        return self.weight.params[x]
    
    def backward(self, output_grad):

        """
        add.at is a nice method that will accumulate repeated
        indexes at their positions:

        Example:

            weight_grad = np.zeros((vocab_size, embed_dim))
            x = np.array([0, 2, 0, 3])          # token indices
            output_grad = np.array([[1,1,1],
                                    [2,2,2],
                                    [3,3,3],
                                    [4,4,4]])     # gradient from loss

            np.add.at(weight_grad, x, output_grad)
            print(weight_grad)

            [[4. 4. 4.]
            [0. 0. 0.]
            [2. 2. 2.]
            [4. 4. 4.]
            [0. 0. 0.]]

        """
        
        ### Initialize Gradients on Weights as Zeros ###
        self.weight.grad = np.zeros_like(self.weight.params)
        
        ### Accumulate Gradients at the Cooresponding Indexes ###
        np.add.at(self.weight.grad, self.x, output_grad)

        return None

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"

class PositionalEmbeddings(GradLayer):
    """
    Learnable positional embeddings added to token embeddings.
    """
    def __init__(self, max_seq_len, embed_dim):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Learnable positional embeddings
        self.weight = GradTensor(
            np.random.normal(
                scale=0.02,
                size=(max_seq_len, embed_dim)
            )
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        Returns x + positional embeddings
        """
        self.seq_len = x.shape[1]
        self.x = x

        # Add positional embeddings (broadcast along batch dimension)
        return x + self.weight.params[:self.seq_len][None, :, :]

    def backward(self, output_grad):
        """
        Gradient flows through both token embeddings (x) and positional embeddings (weight)
        """
        # Gradient w.r.t. positional embeddings
        self.weight.grad = np.zeros_like(self.weight.params)
        np.add.at(self.weight.grad, np.arange(self.seq_len), output_grad.sum(axis=0))

        # Gradient w.r.t. input x
        return output_grad
    
    def __repr__(self):
        return f"PositionalEmbeddings(max_seq_len={self.max_seq_len}"

class LayerNorm(GradLayer):
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = GradTensor(np.ones(self.num_features))
        self.beta = GradTensor(np.zeros(self.num_features))

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input of shape (B, S, E)
        Returns:
            out: normalized and affine-transformed output
        """
        self.x = x  # save input
   
        # Compute mean and variance across features (axis=-1)
        self.mean = np.mean(x, axis=-1, keepdims=True)   # shape (B, S, 1)
        self.var  = np.var(x, axis=-1, keepdims=True)    # shape (B, S, 1)

        # Normalize
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)

        # Scale and shift
        out = self.gamma.params * self.x_hat + self.beta.params

        return out
    
    def backward(self, output_grad):

        # Gradient w.r.t gamma and beta (affine transform)
        self.gamma.grad = np.sum(output_grad * self.x_hat, axis=(0, 1))
        self.beta.grad  = np.sum(output_grad, axis=(0, 1))              

        # Gradient w.r.t normalized input
        dx_hat = output_grad * self.gamma.params  # (B, S, E)
        var_eps = self.var + self.eps

        # LayerNorm backward formula (vectorized)
        mean1 = np.mean(dx_hat, axis=-1, keepdims=True)
        mean2 = np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
        dx = (dx_hat - mean1 - self.x_hat * mean2) / np.sqrt(var_eps)

        return dx
    
class MultiHeadAttention(GradLayer):
    """
    Multi-Head Self-Attention.
    """
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = Linear(self.embed_dim, self.embed_dim)
        self.k_linear = Linear(self.embed_dim, self.embed_dim)
        self.v_linear = Linear(self.embed_dim, self.embed_dim)
        self.out_linear = Linear(self.embed_dim, self.embed_dim)
        self.softmax = SoftMax()

    def forward(self, x, attention_mask=None):
        """
        For self-attention, query = key = value = x (batch_size, seq_len, embed_dim)
        mask: optional (batch_size, 1, seq_len, seq_len) with -inf where masked, 0 otherwise.
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for linear layers: (batch_size * seq_len, embed_dim)
        x_flat = x.reshape(batch_size * seq_len, self.embed_dim)

        # Linear projections
        q = self.q_linear.forward(x_flat)
        k = self.k_linear.forward(x_flat)
        v = self.v_linear.forward(x_flat)
        
        # Reshape back to (batch_size, seq_len, embed_dim) then to heads
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        ### Mask out any non causal positions ###
        self.attention_mask = attention_mask
        if self.attention_mask is not None:
            scores += self.attention_mask
        
        # Apply softmax
        scores_reshaped = scores.reshape(batch_size * self.num_heads, seq_len, seq_len)
        probs = self.softmax.forward(scores_reshaped)
        probs = probs.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        self.probs = probs
        self.q = q
        self.k = k
        self.v = v
        
        # Attention output
        attn = np.matmul(probs, v)
        
        # Concat heads: (batch, seq, embed_dim)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Reshape for out_linear
        attn_flat = attn.reshape(batch_size * seq_len, self.embed_dim)
        out = self.out_linear.forward(attn_flat)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        
        return out

    def backward(self, output_grad):
        """
        Backward through multi-head attention.
        Returns grad w.r.t. query, but since self-attn assumes query=key=value, we sum gradients.
        """
        batch_size, seq_len, _ = output_grad.shape
        
        # Back through out_linear
        output_grad_flat = output_grad.reshape(batch_size * seq_len, self.embed_dim)
        grad_attn_flat = self.out_linear.backward(output_grad_flat)
        grad_attn = grad_attn_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        ### Backward through attn = probs @ v ###
        ### If Y = XW, dL/dW = X^T(dL/dY) and dL/dX = (dL/dY)W^T ###
        ### This was how our linear layer worked, the same idea applied here! ###
        grad_probs = np.matmul(grad_attn, self.v.transpose(0, 1, 3, 2))
        grad_v = np.matmul(self.probs.transpose(0, 1, 3, 2), grad_attn)
        
        # Back through softmax
        grad_probs_reshaped = grad_probs.reshape(batch_size * self.num_heads, seq_len, seq_len)
        grad_scores_reshaped = self.softmax.backward(grad_probs_reshaped)
        grad_scores = grad_scores_reshaped.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        ### 0 out grads from non causal positions ###
        if self.attention_mask is not None:
            grad_scores = np.where(self.attention_mask==-np.inf, 0, grad_scores)

        # Back through scaling
        grad_scores /= math.sqrt(self.head_dim)
        
        ### Backward through scores = q @ k.T ###
        ### Just like before lets first do dL/dQ = dL/dS (k^T)^T = dL/dS (k) ###
        grad_q = np.matmul(grad_scores, self.k)
        ### dL/dK^T = Q^T dL/dS, but we need in terms of dL/dK to continue backprop ###
        ### to get our shapes correct. So dL/dK = [Q^T dL/dS]^T = (dL/dS)^T Q ###
        grad_k = np.matmul(grad_scores.transpose(0, 1, 3, 2), self.q)
        
        # Transpose back
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Reshape for linear layers
        grad_q_flat = grad_q.reshape(batch_size * seq_len, self.embed_dim)
        grad_k_flat = grad_k.reshape(batch_size * seq_len, self.embed_dim)
        grad_v_flat = grad_v.reshape(batch_size * seq_len, self.embed_dim)
        
        # Back through linears
        grad_query = self.q_linear.backward(grad_q_flat)
        grad_key = self.k_linear.backward(grad_k_flat)
        grad_value = self.v_linear.backward(grad_v_flat)
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        grad_query = grad_query.reshape(batch_size, seq_len, self.embed_dim)
        grad_key = grad_key.reshape(batch_size, seq_len, self.embed_dim)
        grad_value = grad_value.reshape(batch_size, seq_len, self.embed_dim)
        
        return grad_query + grad_key + grad_value

class FFN(GradLayer):

    def __init__(self, embed_dim, dim_mult=4):

        self.linear1 = Linear(embed_dim, embed_dim*dim_mult)
        self.relu = ReLU()
        self.linear2 = Linear(embed_dim*dim_mult, embed_dim)
    
    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape
        
        ### Flatten x from (B x S x E) -> (B*S x E) as my linear doesnt support multidimensions
        x_flat = x.reshape(batch_size*seq_len, embed_dim)
        x_flat = self.linear1.forward(x_flat)
        x_flat = self.relu.forward(x_flat)
        x_flat = self.linear2.forward(x_flat)
        
        ### Reshape X back ###
        x = x_flat.reshape(batch_size, seq_len, embed_dim)

        return x
    
    def backward(self, output_grad):

        batch_size, seq_len, embed_dim = output_grad.shape

        ### Flatten output_grad back to (B*S, E) shape ###
        output_grad_flat = output_grad.reshape(batch_size*seq_len, embed_dim)

        ### Compute Backward Grads ###
        output_grad_flat = self.linear2.backward(output_grad_flat)
        output_grad_flat = self.relu.backward(output_grad_flat)
        output_grad_flat = self.linear1.backward(output_grad_flat)

        ### Reshape Back ###
        grad = output_grad_flat.reshape(batch_size, seq_len, embed_dim)

        return grad

class TransformerBlock(GradLayer):
    def __init__(self, embed_dim, num_heads, dropout_p, dim_mult):

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.dim_mult = dim_mult

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FFN(embed_dim, dim_mult)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

    def forward(self, x, attention_mask=None):

        ### Attention + Residual and LayerNorm ###
        attn = self.attention.forward(x, attention_mask)
        attn = self.dropout1.forward(attn)
        x = x + attn
        x = self.norm1.forward(x)

        ### Feedforward + Residual and LayerNorm ###
        ff_out = self.ff.forward(x)
        ff_out = self.dropout2.forward(ff_out)
        x = x + ff_out
        x = self.norm2.forward(x)

        return x
    
    def backward(self, output_grad):

        ### Backward through LayerNorm2 (This is our grad for residual) ###
        grad = self.norm2.backward(output_grad)

        ### Backward through Dropout2 ###
        grad_drop = self.dropout2.backward(grad)

        ### Backward through FeedForward ###
        grad_ff = self.ff.backward(grad_drop)

        ### Add Residual Gradient ###
        grad = grad + grad_ff

        ### Backward through Layernorm1 (This is our grad for residual) ###
        grad = self.norm1.backward(grad)

        ### Backward through Dropout1 ###
        grad_drop = self.dropout1.backward(grad)

        ### Backward through Attention ###
        grad_attn = self.attention.backward(grad_drop)

        ### Residual Connection ###
        grad = grad + grad_attn

        return grad
    
    def __repr__(self):
        return f"TransformerBlock(embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout_p={self.dropout_p}, dim_mult={self.dim_mult})"

class FlattenForLLM(Operation):
    def forward(self, x):
        self.input_shape = x.shape
        batch_size, seq_len, embed_dim = self.input_shape
        return x.reshape(batch_size*seq_len, embed_dim)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def __repr__(self):
        return "FlattenForLLM()"

class NeuralNetwork:
    """
    The most basic Neural Network Ever!
    """
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        return self.forward(input)
    
    # def forward(self, input):
    #     for layer in self.layers:
    #         input = layer.forward(input)
    #     return input

    def forward(self, input, attention_mask=None):
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                input = layer.forward(input, attention_mask)
            else:
                input = layer.forward(input)
        return input

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, GradLayer):
                layer_parameters = layer.parameters()
                params.extend(layer_parameters)
        return params
    
    def train(self):
        """Set network to training mode."""
        self.training = True
        for layer in self.layers:
            self._set_mode_recursive(layer, True)

    def eval(self):
        """Set network to evaluation mode."""
        self.training = False
        for layer in self.layers:
            self._set_mode_recursive(layer, False)
    
    def _set_mode_recursive(self, layer, mode):
        if hasattr(layer, "training"):
            layer.training = mode
        if isinstance(layer, (GradLayer, Operation)):
            for attr_name, attr_value in layer.__dict__.items():
                if isinstance(attr_value, (GradLayer, Operation)):
                    self._set_mode_recursive(attr_value, mode)

    def __repr__(self):
        model_repr = "NeuralNetwork(\n"
        for layer in self.layers:
            model_repr += f"  {repr(layer)}\n"
        model_repr += ")"
        return model_repr
    