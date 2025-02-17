import torch
import torch.nn as nn

class FFNNLM(nn.Module):
    def _init_(self, vocab_size, embed_dim, hidden_dim, context_size, dropout=0.5):
        super()._init_()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embeds = self.embed(x).view(x.size(0), -1)
        out = torch.relu(self.fc1(embeds))
        out = self.dropout(out)
        out = self.fc2(out)
        return torch.log_softmax(out / 0.7, dim=1)

class RNNLM(nn.Module):
    def _init_(self, vocab_size, embed_dim, hidden_dim, dropout=0.5):
        super()._init_()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden):
        embeds = self.embed(x)
        out, hidden = self.rnn(embeds, hidden)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return torch.log_softmax(out / 0.7, dim=1), hidden
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.rnn.hidden_size).to(device)

class LSTMLM(nn.Module):
    def _init_(self, vocab_size, embed_dim, hidden_dim, dropout=0.5):
        super()._init_()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden):
        embeds = self.embed(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return torch.log_softmax(out / 0.7, dim=1), hidden
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(device))