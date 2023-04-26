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