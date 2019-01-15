import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, char):
        embeds = self.word_embeddings(char)
        embeds = embeds.view(1,1, 128)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(1,128)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores