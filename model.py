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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model #dimension of each embedding vector
        self.seq_len = seq_len #maximum sequence length, this is needed to create positional encodings
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)

        # Create a cectore of shape(seq_len) to hold the position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (seq_len, 1) 
        # Unsqueeze to make it a column vector, before that it was a row vector of shape (seq_len,).
        # We need it to be a column vector to perform broadcasting in the next step. 
        
        # Compute the positional encodings using sine and cosine functions

        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        """ expontent term for the denominator in the PE formula, 
            arange create a vector of even indices from 0 to d_model-1, converted to float 
            original formula: 10000^(2i/d_model)
            log(10000) * (2i/d_model) = log(10000) * 2i / d_model because log(a^b) = b*log(a)
            so to get back to the original formula we need to do exp( log(10000) * 2i / d_model )
            a = 10000, b = 2i/d_model
            '-' because in the formula it is in the denominator, we take negative here. 
            so now the formula becomes exp( log(10000) * -2i / d_model ).
        """

        #apply the sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Shape: (seq_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # Shape: (seq_len, d_model/2)

        # Add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

        self.register_buffer('pe', pe)  # Register pe as a buffer so it's not considered a model parameter
        #buffer is a tensor that is not a parameter, so it won't be updated during training, but will be saved and loaded with the model.


    def forward(self, x):
        """
        x is the input tensor of shape (batch_size, sequence_length, d_model)
        We add the positional encodings to the input embeddings to give the model information about the position of each token in the sequence.
        """
        x = x + (self.pe[:, :x.size(1), :]).requires_grad(False)  # Add positional encodings to input embeddings
        """ 
        self.pe has shape (1, seq_len, d_model)
        x has shape (batch_size, sequence_length, d_model) 
        : means we take all the elements in batch dimension (which is 1 here)
        :x.size(1) means we take elements from 0 to sequence_length in the sequence dimension
        : means we take all the elements in the d_model dimension
        That means we are taking positional encodings for the first 'sequence_length' positions only for each batch.
        """
        return self.dropout(x)