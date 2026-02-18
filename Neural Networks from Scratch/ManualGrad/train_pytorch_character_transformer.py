import torch
import torch.nn as nn
import torch.optim as optim
import requests
from tqdm import tqdm
import numpy as np  # For data preprocessing only

# 1. Load dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text)} characters")

# 2. Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = np.array([char_to_idx[ch] for ch in text])

# 3. Batch generation
def get_batch(data, batch_size, seq_len):
    idx = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    inputs = np.stack([data[i:i+seq_len] for i in idx])
    targets = np.stack([data[i+1:i+seq_len+1] for i in idx])
    return torch.from_numpy(inputs).long().cuda(), torch.from_numpy(targets).long().cuda()

# 4. Define model
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers, dropout_p, dim_mult):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * dim_mult,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)
        self.flatten = nn.Flatten(start_dim=0, end_dim=1)  # For LLM output
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, src, causal_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, tgt_mask=causal_mask)
        output = self.flatten(output)  # (batch_size, seq_len, embed_dim) -> (batch_size * seq_len, embed_dim)
        output = self.linear(output)   # (batch_size * seq_len, vocab_size)
        return output

# 5. Initialize model, loss, and optimizer
seq_len = 256
model = TransformerModel(
    vocab_size=vocab_size,
    embed_dim=384,
    max_seq_len=256,
    num_heads=6,
    num_layers=6,
    dropout_p=0.1,
    dim_mult=4
).cuda()
causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# 6. Training loop
model.train()
train_iterations = 5000
batch_size = 32

for epoch in tqdm(range(train_iterations)):
    inputs, targets = get_batch(data, batch_size, seq_len)
    
    # Forward
    logits = model(inputs, causal_mask)
    loss = loss_fn(logits, targets.view(-1))  # Flatten targets to (batch_size * seq_len,)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == targets.view(-1)).float().mean().item() * 100
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# 8. Inference
model.eval()
seed = torch.tensor([[char_to_idx['h']]], dtype=torch.long).cuda()
generated = [seed.item()]

with torch.no_grad():
    for _ in range(seq_len):
        curr_len = seed.size(1)
        causal_mask = torch.triu(torch.ones(curr_len, curr_len) * float('-inf'), diagonal=1).cuda()
        logits = model(seed, causal_mask)
        last_logits = logits[-1]  # Last position logits
        
        # Softmax and sampling
        probs = torch.softmax(last_logits, dim=-1).cpu().numpy()
        next_token = np.random.choice(vocab_size, size=1, p=probs)[0]
        generated.append(next_token)
        seed = torch.tensor([generated], dtype=torch.long).cuda()

print("".join([idx_to_char[i] for i in generated]))