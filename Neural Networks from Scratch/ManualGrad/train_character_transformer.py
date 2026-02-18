"""
Training a tiny GPT type model very close to the MinGPT implementation from Karpathy!
https://github.com/karpathy/minGPT
"""
import cupy as np ### USING NP NOTATION BUT CUPY ALMOST IDENTICAL TO NUMPY
from tqdm import tqdm
import requests
from tqdm import tqdm

import nn
import optim 

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
    return inputs, targets.flatten()

# 4. Build model
model = nn.NeuralNetwork()
model.add(nn.Embedding(vocab_size=vocab_size, embed_dim=384))
model.add(nn.PositionalEmbeddings(max_seq_len=256, embed_dim=384))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.TransformerBlock(embed_dim=384, num_heads=6, dropout_p=0.1, dim_mult=4))
model.add(nn.FlattenForLLM())
model.add(nn.Linear(in_features=384, out_features=vocab_size))

seq_len = 256
causal_mask = np.triu(np.ones((1, 1, seq_len, seq_len)) * -np.inf, k=1)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# 5. Training loop
model.train()
train_iterations = 5000
batch_size = 16

for epoch in tqdm(range(train_iterations)):
    inputs, targets = get_batch(data, batch_size, seq_len)
    
    # # Forward
    logits = model.forward(inputs, causal_mask)
    loss = loss_fn.forward(y_true=targets, logits=logits)
    loss_grad = loss_fn.backward()
    model.backward(loss_grad)
   
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 100 == 0:
        preds = np.argmax(logits, axis=-1)  # predicted indices
        accuracy = np.mean(preds == targets) * 100  # percentage correct
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

        model.eval()
        seed = np.array([[char_to_idx['h']]])  # starting character
        generated = [seed.item()]

        for _ in range(seq_len):
            curr_len = seed.shape[1]
            causal_mask = np.triu(np.ones((1, 1, curr_len, curr_len)) * -1e9, k=1)

            logits = model.forward(seed, causal_mask)  # apply causal mask
            last_logits = logits[-1]  # take logits for last position

            # Softmax and sampling
            probs = np.exp(last_logits - np.max(last_logits))
            probs /= np.sum(probs)

            next_token = np.random.choice(vocab_size, size=1, p=probs)[0].get().item()  # sample 
            generated.append(next_token)
            seed = np.array(generated).reshape(1, -1)
        
        print("".join([idx_to_char[i] for i in generated]))
        model.train()

model.save("work_dir/character_transformer.pkl")
model.load("work_dir/character_transformer.pkl")

