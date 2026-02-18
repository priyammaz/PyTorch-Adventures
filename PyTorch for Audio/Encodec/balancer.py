import torch
from collections import defaultdict

def average_metrics(
    metrics, 
    count=1, 
    accelerator=None
):
    """"
    Average a dictionary of metrics across all distributed processes.
    
    This function computes a weighted average of metrics across all GPUs in a distributed
    training setup. The weighting is determined by the `count` parameter, which typically
    represents the batch size on each GPU. This ensures that GPUs with larger batches have
    proportionally more influence on the final averaged metrics.
    
    The averaging formula for each metric is:
        averaged_metric = sum(metric_i * count_i) / sum(count_i)
    
    where the sum is taken across all processes (GPUs).
    """

    if accelerator is None or accelerator.num_processes == 1:
        return metrics
    
    ### Unpack the dictionary ###
    keys, values = zip(*metrics.items())
    
    ### Create a tensor with metrics values + count ###
    ### values is a list of numbers [v1, v2, v3, ..]
    ### count is a single number, we make it a list [count]
    ### we concat to make [v1, v2, v3, ..., count]
    tensor = torch.tensor(list(values) + [count], dtype=torch.float32, device=accelerator.device)

    ### Sum across all processes ##
    ### gpu0: [v01, v02, v03, ..., count_0]
    ### gpu1: [v11, v12, v13, ..., count_1]
    ### reduces: [v01+v11, v02+v12, v03+v13, ..., count_0+count_1]
    tensor = accelerator.reduce(tensor, reduction="sum")
    
    ### Normalize by total count and convert back to dict ###
    ### [v01+v11 / count_0+count_1, v02+v12 / count_0+count_1, v03+v13 / count_0+count_1, ...,]
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    
    ### Convert back to a dictionary with the values now averaged across GPUs ###
    return dict(zip(keys, averaged))

def averager(beta=1.0, state=None):
    """This is simple EMA (where if beta=1 we are just simple average)"""

    ### Global vars in the function scope that we update ###
    if state is not None:
        fix = defaultdict(float, state.get('fix', {}))
        total = defaultdict(float, state.get('total', {}))
    else:
        fix = defaultdict(float)
        total = defaultdict(float)

    def _update(metrics):

        ### make sure local/fix refer to the global variable so we are constantly updating that! ###
        nonlocal total, fix

        ### For every metric ###
        for key, value in metrics.items():
            
            if value is not None:

                ### Update current metric total: current * beta + new ###
                total[key] = total[key] * beta + float(value)
                
                ### Update normalization: old norm * beta + 1 for the new sample ###
                fix[key] = fix[key] * beta + 1
        
        ### Return the normalized metrics ###
        return {key: (tot / fix[key] if tot is not None else None) for key, tot in total.items()}
    
    def get_state():
        """get the running accumulations we have"""
        return {
            "fix": dict(fix), 
            "total": dict(total)
        }

    def load_state(state):
        """load running accumulations"""
        nonlocal fix, total
        fix = defaultdict(float, state.get('fix', {}))
        total = defaultdict(float, state.get('total', {}))

    ### Attach state management methods to the update method ###
    _update.get_state = get_state
    _update.load_state = load_state

    return _update

class Balancer:
    """

    ### WHATS THE ISSUE ###

    We have a bunch of loss values we computed w.r.t the outputs of the Generator. Lets 
    call them L1, L2, L3 and L4. 

    When we backprop one of the chain rule ops will be Li w.r.t outputs of Generator. And 
    because we have multiple losses, we will have dL1/dO, dL2/dO, dL3/dO, dL4/dO, and in
    standard pytorch fashion, these losses will accumulate (sum together). But before we sum
    them together we have an issue, each of these losses have different magnitudes, and will
    have a different amount of contribution to the overall gradient. 

    But we have some weights. L1: w1, L2: w2, L3: w3, L4: w4. We would like our contributions
    from all these upstream grads to be proportional to the weights we set! And so exactly
    for that reason, we have our Loss Balancer.  

    ### WHAT THE BALANCER DOES ###
    
    Instead of weighting losses directly, we:

    1. Compute gradient norms for each loss (norms tell us the magnitude):
       norm1 = ||dL1/dO||
       norm2 = ||dL2/dO||
       norm3 = ||dL3/dO||
       norm4 = ||dL4/dO||

    2. Track these norms with EMA (to stabilize across batches):
       avg_norm1 = EMA(norm1)
       avg_norm2 = EMA(norm2)
       avg_norm3 = EMA(norm3)
       avg_norm4 = EMA(norm4)

    3. Compute desired ratios from weights:
       total_weight = w1 + w2 + w3 + w4
       ratio1 = w1 / total_weight
       ratio2 = w2 / total_weight
       ratio3 = w3 / total_weight
       ratio4 = w4 / total_weight
       
       (Note: these ratios sum to 1)

    4. Scale each gradient so its norm matches the desired contribution:
       For loss i, we want its contribution to have norm: ratio_i * total_norm
       
       Current norm: avg_norm_i
       Desired norm: ratio_i * total_norm
       Scale factor: (ratio_i * total_norm) / avg_norm_i
       
       scaled_grad_i = grad_i * scale_i
    
    5. Sum the scaled gradients:
       final_grad = scaled_grad1 + scaled_grad2 + scaled_grad3 + scaled_grad4


    ### CAVEAT FOR DISTRIBUTED TRAINING ###
    
    Nothing changes in DDP training, except when we compute our Norms, we need to do
    it over the gradients accumulated from ALL GPUS!

    """

    def __init__(self, 
                 weights, 
                 total_norm=1., 
                 ema_decay=0.999, 
                 epsilon=1e-12,
                 max_scale=100,
                 min_scale=0.01, 
                 accelerator=None):
        
        self.weights = weights
        self.total_norm = total_norm
        self.epsilon = epsilon
        self.accelerator = accelerator
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.averager = averager(ema_decay) # returns a method
        
    def backward(self, losses, input, retain_graph=False):

        norms = {}
        grads = {}

        ### For every loss ###
        for name, loss in losses.items():
            
            ### Compute the gradient of the loss w.r.t the model outputs ### 
            ### which will be passed in as input ###
            grad, *_ = torch.autograd.grad(loss, [input], retain_graph=True, allow_unused=True)
            
            if grad is not None:

                ### The grad of loss w.r.t input will be in the shape of the input ###
                ### which in our case is the output of the model so its (B x 1 x L) ###
                ### so lets compute the per-sample norm of the grads ###
                dims = tuple(range(1, grad.dim())) # all dims after batch dim
                norm = grad.norm(dim=dims)

                ### Average up the norm across the batch ###
                norm = norm.mean()

                ### Store the grad and norm per loss ###
                norms[name] = norm
                grads[name] = grad

                ### What is the per gpu batch size? use the last grad as they are all the same for each loss ###
                count = len(grad)
            
            else:
                norms[name] = None
                grads[name] = None
        
        ### Compute the avg norms across GPUs ###
        avg_norms = average_metrics(self.averager(norms), count, self.accelerator)

        ### Get our total weights and compute ratios ###
        total_weights = sum([self.weights[k] for k in avg_norms.keys()])
        ratios = {k: w / total_weights for (k,w) in self.weights.items()}

        ### Init our output grad ###
        out_grad = 0

        ### Loop through and scale our gradients ###
        for name, avg_norm in avg_norms.items():

            ### Compute scaling constants ###
            scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)

            ### Clip scale incase of extreme values (helps with training at the start) ###
            scale = max(self.min_scale, min(scale, self.max_scale))

            
            if grads[name] is not None:
                
                ### Scale gradients ###
                grad = grads[name] * scale
            
                ### Accumulate grads ###
                out_grad += grad
        
        ### We have now scaled and accumulated up our gradients w.r.t the input ###
        ### Lets keep the backward pass going now from the input (output of encoder) ###
        ### to the rest of the model !! ###
        if self.accelerator is None:
            input.backward(out_grad, retain_graph=retain_graph)
        else:
            self.accelerator.backward(loss=input, # not exactly the loss but backprop should start from the input
                                      gradient=out_grad, 
                                      retain_graph=retain_graph)

    def state_dict(self):
        """save the state dict"""
        return {"averager_state": self.averager.get_state()}

    def load_state(self, state_dict):
        """load the state dict"""
        self.averager.load_state(state_dict["averager_state"])