import torch.nn as nn

class SLSTM(nn.Module):
    def __init__(self, 
                 dimension, 
                 num_layers=2, 
                 bidirectional=True,
                 skip=True):
        
        super().__init__()

        self.lstm = nn.LSTM(dimension, 
                            dimension, 
                            num_layers,
                            bidirectional=bidirectional, 
                            batch_first=False)
        
        ### bidirectional model will double the output dim
        if bidirectional:
            self.proj = nn.Linear(dimension * 2, dimension)
        else:
            self.proj = nn.Identity()

        self.skip = skip

    def forward(self, x):
        
        ### permute x from (b x c x l) -> (l x b x c)
        x = x.permute(2,0,1)
        
        ### pass through lstm ###
        y, _ = self.lstm(x)

        ### Project back if needed ###
        y = self.proj(y)
        
        ### add skip connection ###
        if self.skip:
            y = y + x

        ### Put back how we got it ###
        y = y.permute(1,2,0)

        return y