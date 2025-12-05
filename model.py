import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model #dimension of each embedding vector
        self.vocab_size = vocab_size #size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model) #embedding layer (we will have vocab_size vectors of dimension d_model)

    def forward(self, x): 
        
        """
        x is the input tensor of shape (batch_size, sequence_length) containing token indices.
        So the output will be of shape (batch_size, sequence_length, d_model).
        Each batch will have a sequence of token indices, and each token index will be mapped to its corresponding embedding vector
        """

        return self.embedding(x) *  math.sqrt(self.d_model)


