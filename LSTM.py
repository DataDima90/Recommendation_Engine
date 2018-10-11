import torch
import torch.nn as nn

class LSTMRating(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_items, num_output):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_output)
        self.hidden = self.init_hidden()

    def init_hidden(self):
    	# initialize both hidden layers
        return ((torch.zeros(1, 1, self.hidden_dim)),
                (torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sequence):
        embeddings = self.item_embeddings(sequence)
        output, self.hidden = self.lstm(embeddings.view(len(sequence), 1, -1),
                                        self.hidden)
        rating_scores = self.linear(output.view(len(sequence), -1))
        return rating_scores

    def predict(self, sequence):
        rating_scores = self.forward(sequence)
        return rating_scores
