"""quantization with accelerate DDP support"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator

def ema_inplace(moving_average, new, decay):
    """
    ema update parameter. moving_avg = moving_avg + (1-decay) * new
    """
    moving_average.data.mul_(decay).add_(new, alpha=(1-decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    # https://en.wikipedia.org/wiki/Additive_smoothing
    return (x + eps) / (x.sum() + n_categories * eps)

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples, num):
    """
    samples should be a (N x dim) matrix!
    """
    
    num_samples = samples.shape[0]
    device = samples.device

    ### If we have more vectors we want to sample 
    ### simply random sort and index
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]

    ### If we dont then we will have to random sample with repetition 
    ### but with a sufficient batch size this shouldnt be an issue
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)
    
    return samples[indices]

def kmeans(samples, num_clusters, num_iters):

    """
    This is the standard kmeans algo! We will use this to initialize our 
    embedding matrix, after which its no longer used!
    """

    dim = samples.shape[-1]
    dtype = samples.dtype
    
    ### From the (N x dim) matrix, sample num_clusters worth of
    ### samples to start as our cluster centers. We will be iteratively
    ### update these!
    means = sample_vectors(samples, num_clusters)   

    ### Iterate
    for _ in range(num_iters):

        ### create a tensor that reshapes our samples ###
        ### samples_: (N x dim) -> (N x 1 x dim)
        ### means_: (num_clusters x dim) -> (1 x num_clusters x dim)
        samples_, means_ = samples.unsqueeze(1), means.unsqueeze(0)

        ### diffs between every pairwise vector ###
        ### how far is every cluster center vector from every sample in the dataset ###
        ### diffs: (N x 1 x dim) - (1 x num_clusters x dim) => (N x num_clusters x dim)
        diffs = samples_ - means_

        ### Now we need to wrap up the Euclidean Distance Computation ###
        ### we have the diff along the dim dimension, lets square and add them up! ###
        ### we multiply by -1 as we will find the max negative distance rather than then ###
        ### min distance but its the same thing! I want to just keep it close to the original ###
        ### implementation!
        ### dists: (N x num_clusters)
        dists = -1 * (diffs ** 2).sum(dim=-1)

        ### For each N find the closest (or largest negative distance) cluster
        ### buckets: (N,) -> index of which cluster was the closest to this point
        buckets = dists.max(dim=-1).indices

        ### Now we know the closest cluster center for each, count up how many points were
        ### assigned to each of those clusters
        ### cluster 0 => 3 points
        ### cluster 1 => 1 point
        ### cluster 2 => 2 points
        bins = torch.bincount(buckets, minlength=num_clusters)

        ### Handle Empty Clusters (nothing assigned) ###
        ### we will need to divide, we dont want to divide by zero ###
        zero_mask = (bins == 0)
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        ### Update means ###
        ### Create a new zero tensor (on the same device as the original buckets)
        ### and use this as our accumulator
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)

        ### We want the average of each of these groups to be our new cluster center ###
        ### to compute the average we first need to sum, we can divide after ###
        ### basically for every point assigned to a specific cluster center, add ###
        ### all those vectors together! ###
        ### basically this operation
        ### for i in range(N):
        ###     cluster = buckets[i]
        ###     new_means[cluster] += samples[i]
        ### but for efficiency we can use scatter ops
        ### repeated_buckets: (N,) -> (N,dim) This is so we have the index for every value along the
        ### embed dim so scatter ops can just accumulate it up!
        repeated_buckets = buckets.unsqueeze(-1).repeat(1, dim)
        new_means.scatter_add_(dim=0, index=repeated_buckets, src=samples)
        
        ### We just added we now need to average, so divide by number of samples in every bin 
        ### broadcasting over the embed dim dimension
        new_means = new_means / bins_min_clamped.unsqueeze(-1)

        ### update our means ###
        ### for empty clusters use the old mean, for non empty clusters update ###
        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook w/ DDP support"""
    def __init__(self, 
                 dim, 
                 codebook_size, 
                 kmeans_init=False,
                 kmeans_iter=50, 
                 decay=0.99, 
                 epsilon=1e-5, 
                 threashold_ema_dead_code=2,
                 accelerator=None):
        
        super().__init__()

        self.decay = decay
        self.codebooks_size = codebook_size
        self.kmeans_iters = kmeans_iter
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threashold_ema_dead_code
        self.accelerator = accelerator # we pass this in for DDP purposes
        
        ### What is our init method? ###
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        ### Save as buffers so its saved with the model weights ###
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))  # have we initialized?
        self.register_buffer("cluster_size", torch.zeros(codebook_size)) # How many samples assigned to each cluster?
        self.register_buffer("embed", embed)                             # Store the embeddings
        self.register_buffer("embed_avg", embed.clone())                 # Store a clone for running avg 

    def _is_distributed(self):
        return self.accelerator is not None and self.accelerator.num_processes > 1

    def _broadcast_buffers(self):
        """Sync all buffers across all processes."""
        if not self._is_distributed():
            return
            
        for buffer in self.buffers():
            buffer.data = accelerate.utils.broadcast(buffer.data, from_process=0)

    @torch.jit.ignore
    def init_embed_(self, data):

        if self.inited:
            return 
        
        ### If we are in DDP mode ###
        if self._is_distributed():

            ### We gather all the data from all the gpus, as our data is sharded 
            ### across gpus (in ddp)
            all_data = self.accelerator.gather(data)
            
            ### Only on the main process do Kmeans and update buffers
            if self.accelerator.is_main_process:
                embed, cluster_size = kmeans(all_data, self.codebooks_size, self.kmeans_iters)
                self.embed.data.copy_(embed)
                self.embed_avg.data.copy_(embed.clone())
                self.cluster_size.data.copy_(cluster_size) # computed across all samples across all GPUs
                                                           # so this is already scaled correctly
            
            # Wait for main process to finish
            self.accelerator.wait_for_everyone()

            # Broadcast buffers from main process to all others 
            self._broadcast_buffers()
        
        else:

            embed, cluster_size = kmeans(data, self.codebooks_size, self.kmeans_iters)
            self.embed.data.copy_(embed)
            self.embed_avg.data.copy_(embed.clone())
            self.cluster_size.data.copy_(cluster_size)
        
        ### change flag as we have initialized
        self.inited.data.copy_(torch.Tensor([True]))

        ### Ensure flag is synced ###
        if self._is_distributed():
            self.inited.data = accelerate.utils.broadcast(self.inited.data, from_process=0)

    def replace_(self, samples, mask):
        
        ### Method to replace codes our self.embeds based on a mask (and resampling) ###
        modified_codebook = torch.where(
            mask.unsqueeze(-1), sample_vectors(samples, self.codebooks_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)
    
    def expire_codes_(self, batch_samples):
        """batch_samples is accumulated across GPUS incase we are in DDP"""

        ### If 0 there is nothing to do 
        if self.threshold_ema_dead_code == 0:
            return
        
        ### expired codes dont have enough sample assigned to them! ###
        ### remeber self.cluster_size has already been synced so its the ###
        ### same across all GPUS ###
        expired_codes = (self.cluster_size < self.threshold_ema_dead_code)

        if not torch.any(expired_codes):
            return
        
        ### Data is (B x L x E), we can flatten to (N=B*L, E) in a bunch of ways
        ### we will just use einops this time!
        batch_samples = einops.rearrange(batch_samples, "... d -> (...) d")

        ### Only main process decides which codes to replace and doesreplacement
        if not self._is_distributed() or self.accelerator.is_main_process:
            ### replace our expired codes in codebook with a new code ###
            ### only do this on the main process
            self.replace_(batch_samples, mask=expired_codes)

        ### Wait for main process to finish
        if self._is_distributed():
            self.accelerator.wait_for_everyone()
        
        ### Broadcast updated embeddings to all processes ###
        if self._is_distributed():
            self.embed.data = accelerate.utils.broadcast(self.embed.data, from_process=0)

    def preprocess(self, x):
        """flatten all but last dims"""
        x = einops.rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):

        ### (N x E) -> (E x N)
        embed = self.embed.t()

        ### compute distance between each x and embeds
        ### x**2 + 2xe + e**2
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        ### find the closest (largest negative) embedding for each x
        embed_ind = dist.max(dim=-1).indices

        return embed_ind
    
    def postprocess_emb(self, embed_ind, shape):
        """reshape to the original input shape (except embedding dim)"""
        return embed_ind.reshape(*shape[:-1])
     
    def dequantize(self, embed_ind):
        """given indexes grab them from the embed matrix"""
        return F.embedding(embed_ind, self.embed)

    @torch.no_grad()
    def encode(self, x):
        
        # B x L x E
        shape = x.shape
        
        # N x E, where N = B*L
        x = self.preprocess(x)

        # N
        embed_ind = self.quantize(x)

        # B * L
        embed_ind = self.postprocess_emb(embed_ind, shape)

        return embed_ind
    
    @torch.no_grad()
    def decode(self, embed_ind):
        return self.dequantize(embed_ind)
    
    def forward(self, x):
        """forward should only be used during training time as it updates embeddings!"""
        ### Store original data attributes
        shape, dtype = x.shape, x.dtype

        ### preprocess data 
        x = self.preprocess(x)

        ### In the first forward pass we init our codebook 
        ### This will trigger the flag self.inited and become a
        ### no-op afterwards
        self.init_embed_(x)

        ### Quantize 
        ### indexes of what codevector is assigned to each input (N,)
        embed_ind = self.quantize(x) 

        ### OHE version of the same thing (N,K)
        embed_onehot = F.one_hot(embed_ind, self.codebooks_size).type(dtype)

        ### Postprocess the embed_ind
        embed_ind = self.postprocess_emb(embed_ind, shape)

        ### Quantized values
        quantized = self.dequantize(embed_ind)

        ### If in training mode we update the codebook 
        if self.training:

            # Gather batch samples from all GPUs for code expiration computation
            if self._is_distributed():
                batch_samples = self.accelerator.gather(x)
            else:
                batch_samples = x
            
            # update any expired codes
            self.expire_codes_(batch_samples)

            ### Compute local statistics on this GPU
            local_cluster_count = embed_onehot.sum(0)

            ### Compute the embed_sum which computes the 
            ### sum of all vectors assigned to a cluster
            ### (dim x N) @ (N x K) -> (dim x K)
            local_embed_sum = x.t() @ embed_onehot

            ### If distributed add across GPUs ###
            if self._is_distributed():
                global_cluster_count = self.accelerator.reduce(local_cluster_count, reduction="sum")
                global_embed_sum = self.accelerator.reduce(local_embed_sum, reduction="sum")
            else:
                global_cluster_count = local_cluster_count
                global_embed_sum = local_embed_sum
            
            # Update EMA with global statistics (summed across all GPUs)
            # this will happen on all GPUs, but we will just copy the main gpus results 
            # into the other ones later so it doesnt really matter!
            ema_inplace(self.cluster_size, global_cluster_count, self.decay) # update cluster size
            ema_inplace(self.embed_avg, global_embed_sum.t(), self.decay) # update running stats of embeds

            # Update embeddings (same computation on all ranks using global stats)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebooks_size, self.epsilon)
                * self.cluster_size.sum()
            )

            ### Normalize to get our new codebook vectors ###
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # Sync all buffers across GPUs to ensure consistency (could be small numerical inconsistencies)
            if self._is_distributed():
                self._broadcast_buffers()
            
            if self._is_distributed():
                self.accelerator.wait_for_everyone()
        
        return quantized, embed_ind

class VectorQuantization(nn.Module):
    """standard VQ method"""
    def __init__(self, 
                 dim, 
                 codebook_size, 
                 codebook_dim=None, 
                 decay=0.99, 
                 epsilon=1e-5,
                 kmeans_init=True, 
                 kmeans_iters=50, 
                 threshold_ema_dead_code=2, 
                 commitment_weight=1,
                 accelerator=None):
        
        super().__init__()

        self.decay = decay
        self.codebooks_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.accelerator = accelerator # we pass this in for DDP purposes
        self.commitment_weight = commitment_weight

        ### Setup codebook proj ###
        codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = codebook_dim != dim
        self.proj_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.proj_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

        ### Setup codebook ###
        self._codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size, 
                                           kmeans_init=kmeans_init, kmeans_iter=kmeans_iters,
                                           decay=decay, epsilon=epsilon, 
                                           threashold_ema_dead_code=threshold_ema_dead_code, 
                                           accelerator=accelerator)
    
    @torch.no_grad()
    def encode(self, x):
        x = einops.rearrange(x, "b d n -> b n d")
        x = self.proj_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in
    
    @torch.no_grad()
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.proj_out(quantize)
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize
    
    def forward(self, x):
        """forward should only be used during training time as it updates embeddings!"""
        device = x.device

        # Set correct device
        x = einops.rearrange(x, "b d n -> b n d")

        # Project to codebook dim
        x = self.proj_in(x)

        # quantize
        quantize, embed_ind = self._codebook(x)

        # stop gradient 
        if self.training:
            quantize = x + (quantize - x).detach()
        
        # Placeholder
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            
            if self.commitment_weight > 0:
                ### We want our encoder to move towards the codebook values
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight
        
        ### Project and return ###
        quantize = self.proj_out(quantize)
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss

class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers, **kwargs):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    @torch.no_grad()
    def encode(self, x, n_q=None):
        
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        out_indices = torch.stack(all_indices)
        return out_indices

    @torch.no_grad()
    def decode(self, q_indices):

        quantized_out = torch.tensor(0.0, device=q_indices.device)

        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
            
        return quantized_out
    
    def forward(self, x, n_q=None):
        """forward should only be used during training time as it updates embeddings!"""
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:

            ### Pass through a VQ
            quantized, indices, loss = layer(residual)

            ### Compute the residual (diff from true and estimate)
            residual = residual - quantized.detach()

            ### Accumulate quantized out 
            quantized_out = quantized_out + quantized

            ### Store 
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices)
        
        return quantized_out, out_indices, out_losses

def test_kmeans():
    """helper method to just test our kmeans out!"""

    import matplotlib.pyplot as plt

    true_centers = torch.tensor([
        [-5.0, -5.0],
        [0.0, 5.0],
        [5.0, -2.0]
    ])

    num_points_per_cluster = 300
    dim = 2
    samples_list = []

    for center in true_centers:
        points = center + torch.randn(num_points_per_cluster, dim)
        samples_list.append(points)

    # combine all clusters
    samples = torch.cat(samples_list, dim=0)

    means, bins = kmeans(samples, 3, 20)

    plt.figure(figsize=(6,6))

    # plot samples
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5)

    # plot learned centers
    plt.scatter(means[:, 0], means[:, 1],
                color="red",
                s=200,
                marker="x",
                linewidths=3)

    plt.title("K-means clustering result")
    plt.show()


def test_codebook():
    accelerator = Accelerator()
    
    # Test parameters
    dim = 128
    codebook_size = 16
    seq_len = 100
    batch_size = 8
    
    # Create codebook
    codebook = EuclideanCodebook(
        dim=dim,
        codebook_size=codebook_size,
        kmeans_init=True,
        accelerator=accelerator,
        threashold_ema_dead_code=10
    )
    
    # Move to device and prepare
    codebook = codebook.to(accelerator.device)
    codebook.train()
    
    accelerator.print(f"\n{'='*50}")
    accelerator.print(f"Testing EuclideanCodebook with DDP")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Current process: {accelerator.process_index}")
    accelerator.print(f"{'='*50}\n")
    
    # Test initialization
    accelerator.print("Step 1: Testing k-means initialization...")
    x = torch.randn(batch_size, seq_len, dim).to(accelerator.device)
    
    # First forward pass (triggers k-means init)
    quantized, indices = codebook(x)
    
    accelerator.wait_for_everyone()
    

    # Check if embeddings are synced across GPUs
    if accelerator.is_main_process:
        embed_sum = codebook.embed.sum().item()
        cluster_size_sum = codebook.cluster_size.sum().item()
        accelerator.print(f"✓ Initialization complete")
        accelerator.print(f"  Embed sum: {embed_sum:.4f}")
        accelerator.print(f"  Cluster size sum: {cluster_size_sum:.4f}")
    
    accelerator.wait_for_everyone()
    
    # Test multiple training steps
    accelerator.print("\nStep 2: Testing training updates...")
    for step in range(3):
        x = torch.randn(batch_size, dim).to(accelerator.device)
        quantized, indices = codebook(x)
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            embed_sum = codebook.embed.sum().item()
            cluster_size_sum = codebook.cluster_size.sum().item()
            accelerator.print(f"  Step {step+1}: embed_sum={embed_sum:.4f}, cluster_size_sum={cluster_size_sum:.4f}")
    
    # Verify synchronization across GPUs
    accelerator.print("\nStep 3: Verifying buffer synchronization...")
    
    # Gather embeddings from all processes
    if accelerator.num_processes > 1:
        embed_gathered = accelerator.gather(codebook.embed)
        cluster_size_gathered = accelerator.gather(codebook.cluster_size)
        
        if accelerator.is_main_process:

            # Check if all processes have identical embeddings
            embed_split = embed_gathered.view(accelerator.num_processes, codebook_size, dim)
            cluster_size_split = cluster_size_gathered.view(accelerator.num_processes, codebook_size)
            
            embed_diff = (embed_split[0].unsqueeze(0) - embed_split[1:]).abs().max().item()
            cluster_diff = (cluster_size_split[0].unsqueeze(0) - cluster_size_split[1:]).abs().max().item()
            
            accelerator.print(f"Max embedding difference across GPUs: {embed_diff:.2e}")
            accelerator.print(f"Max cluster_size difference across GPUs: {cluster_diff:.2e}")
            
            if embed_diff < 1e-5 and cluster_diff < 1e-5:
                accelerator.print("SUCCESS: All buffers are properly synchronized!")
            else:
                accelerator.print("FAILED: Buffers are not synchronized!")
    else:
        accelerator.print("Single process mode - no synchronization needed")
    
    accelerator.print(f"\n{'='*50}\n")
    
    accelerator.end_training()

if __name__ == "__main__":
    # test_kmeans()
    test_codebook()