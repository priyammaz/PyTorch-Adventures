import os
import cupy as np
import math

##### QUICK ENABLE FOR TENSOR CORE OPS ###
device = np.cuda.Device()
# string containing the major index and the minor index. 
# For example, compute capability 3.5 is represented by the string ‘35’.
cc_major, cc_minor = device.compute_capability 
if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"
##########################################

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
                size=(in_features, out_features),
                dtype=np.float32
            )
        )

        if bias:
            self.bias = GradTensor(
                np.zeros((1, out_features), dtype=np.float32)
            )
        else:
            self.bias = None

    def forward(self, x):
        self.x = x
        out = np.matmul(x, self.weight.params)
        if self.bias is not None:
            out += self.bias.params
        return out

    def backward(self, grad_output):
        # Grad w.r.t weight
        self.weight.grad = np.matmul(self.x.T, grad_output)
        
        # Grad w.r.t bias
        if self.bias is not None:
            self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)
        
        # Grad w.r.t input
        return np.matmul(grad_output, self.weight.params.T)
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

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

class ReLU(Operation):
    def forward(self, x):
        self.x = x
        x = np.clip(x, a_min=0, a_max=None)
        return x
    
    def backward(self, output_grad):
        grad = np.zeros_like(self.x)
        grad[self.x > 0] = 1
        output_grad = grad * output_grad
        return output_grad
    
    def __repr__(self):
        return "ReLU()"

class Embedding(GradLayer):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weight = GradTensor(
            np.random.normal(
                scale=0.02,
                size=(vocab_size, embed_dim),
                dtype=np.float32
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
                size=(max_seq_len, embed_dim),
                dtype=np.float32
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
        self.gamma = GradTensor(np.ones(self.num_features, dtype=np.float32))
        self.beta = GradTensor(np.zeros(self.num_features, dtype=np.float32))

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
            self.mask = self.mask.astype(np.float32)

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
    