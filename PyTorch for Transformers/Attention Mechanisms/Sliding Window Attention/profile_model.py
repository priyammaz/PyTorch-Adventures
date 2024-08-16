import torch
import argparse
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt

from utils import RobertaConfig
from model import RobertaLocalAttention

parser = argparse.ArgumentParser("Arguments for RoBERTa Model Profiling")

parser.add_argument(
    "--num_attention_heads",
    help="Number of heads of attention",
    default=12,
    type=int
)

parser.add_argument(
    "--embedding_dimension",
    help="Transformer embedding dimension",
    default=768,
    type=int
)

### WINDOWED ATTENTION CONFIG ###
parser.add_argument(
    "--context_length",
    help="What is the maximum sequence length we want to use?",
    default=2048
)

parser.add_argument(
    "--window_size",
    help="What window size do you want to use for windowed attention?",
    default=512, 
    type=int
)

parser.add_argument(
    "--look_backward",
    help="how many previous windows do we want to attend to?",
    default=1, 
    type=int
)

parser.add_argument(
    "--look_forward",
    help="how many future windows do we want to attend to?",
    default=1, 
    type=int
)

### Compute Arguments ###
args = parser.parse_args()

### Grab Config ###
config = RobertaConfig(


    embedding_dimension = args.embedding_dimension,
    num_attention_heads = args.num_attention_heads,
    window_size=args.window_size, 
    look_backward=args.look_backward, 
    look_forward=args.look_forward,
    context_length = args.context_length,

)

class SelfAttention(nn.Module):
    """
    I made a quick implementation of SelfAttention here as my RoBERTa implementation 
    uses flash attention, it isnt really fair to compare custom cuda kernels to 
    my definitely unoptimized windowed attention. 
    """
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        
        ### Store Config ###
        self.config = config
        
        ### Sanity Checks ###
        assert config.embedding_dimension % config.num_attention_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.head_dim = config.embedding_dimension // config.num_attention_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        

    def forward(self, x):

        ### Store Shape ###
        batch, seq_len, embed_dim = x.shape

        ### Compute Attention with Flash Attention ###
        q = self.q_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        
        attn = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attention_out = attn @ v

        ### Compute Output Projection ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

### Load Different Attention Mechanisms so we can Compute Flops Comparison ###
self_attention = SelfAttention(config).to("cuda")
local_attention = RobertaLocalAttention(config).to("cuda")

lens = [args.window_size*(i+1) for i in range(40)]
self_attn_flops = []
local_attn_flops = []

for len in lens:
    rand = torch.randn((1,len,768), device="cuda")
    sa_flops = FlopCountAnalysis(self_attention, rand).total()
    la_flops = FlopCountAnalysis(local_attention, rand).total()

    self_attn_flops.append(sa_flops)
    local_attn_flops.append(la_flops)

plt.plot(lens, self_attn_flops, label="Self Attention")
plt.plot(lens, local_attn_flops, label="Local Attention")
plt.xlabel("Sequence Length")
plt.ylabel("FLOPS")
plt.title("FLOPS Comparison Between Full and Windowed Attention")
plt.legend()
plt.savefig("flops_comparison.png", dpi=200)


