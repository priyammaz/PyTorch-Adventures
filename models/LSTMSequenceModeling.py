import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, device):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        ### Define Embedding Matrix ###
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        ### Define LSTM ###
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        ### Final Classifier Layers ###
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        batch_size, sequence_len = x.shape
        embeddings = self.embedding(x)  # (Batch x Sequence Len x Embedding Dim)

        ### INITIALIZE HIDDEN AND CELL STATE AS 0 ###
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device)  # Num Layers x Batch Size x Hidden State
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device)  # Num Layers x Batch Size x Hidden State

        ### PASS THROUGH LSTM BLOCK ###
        output, (hn, cn) = self.lstm(embeddings, (h0, c0))

        # Output -> [batch x seqlen x hidden]
        # Hn -> [num_layers x batch x hidden]
        # Cn -> [num_layers x batch x hidden]

        ### CUT OFF LAST HIDDEN STATE FOR EVERY BATCH ###
        last_hidden = output[:, -1, :]  # Batch Size x Hidden

        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out
    
class LSTMForGeneration(nn.Module):
    def __init__(self, embedding_dim, num_characters, hidden_size, n_layers, device, char2idx, idx2char):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_characters = num_characters
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.char2idx = char2idx
        self.idx2char = idx2char

        self.embedding = nn.Embedding(self.num_characters, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_characters)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        ### INITIALIZE HIDDEN AND CELL STATE AS 0 ###
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            self.device)  # Num Layers x Batch Size x Hidden State
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            self.device)  # Num Layers x Batch Size x Hidden State

        output, (hn, cn) = self.lstm(x, (h0, c0))
        # Output -> [batch x seqlen x hidden]
        # Hn -> [num_layers x batch x hidden]
        # Cn -> [num_layers x batch x hidden]

        out = self.fc(output)  # Batch x Seq Len X Num Characters

        ### Out has a final dimension of num characters where each set was generated with information from the previous token ###
        return out

    def write(self, text, max_characters, train=True):
        idx = torch.tensor([self.char2idx[c] for c in text]).to(self.device)

        hidden = torch.zeros(self.n_layers, self.hidden_size).to(self.device)  # Num Layers x Hidden State
        cell = torch.zeros(self.n_layers, self.hidden_size).to(self.device)  # Num Layers x Hidden State

        for i in range(max_characters):
            if i != 0:
                # 1 x Embedding
                selected_idx = idx[-1].unsqueeze(
                    0)  # After the first iteration, we use the last predicted char to predict the next one

            else:
                # Seq x Embedding
                selected_idx = idx  # In the first iteration, we want to build up the hidden and cell state with the input chars

            x = self.embedding(selected_idx)
            out, (hidden, cell) = self.lstm(x, (hidden, cell))
            out = self.fc(out)  # Seq_len x num_characters

            if len(out) > 1:  # In the first iteration, we use all the character to build the H and C but we only use the last token for prediction
                out = out[-1, :].unsqueeze(0)

            probs = self.softmax(out)  # Take softmax along character dimension to convert to probability vector

            if train:
                idx_next = torch.multinomial(probs,
                                             num_samples=1)  # Sample from Multinomial distribution to get next index
            else:
                idx_next = torch.argmax(probs)

            idx = torch.cat([idx, idx_next[0]])  # concatenate the next index to our original index vector

        gen_string = "".join([self.idx2char[int(c)] for c in idx])
        return gen_string