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
    

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(d_model))   # Shift parameter

    def forward(self, x):
        """
        x is the input tensor of shape (batch_size, sequence_length, d_model)
        We normalize the input tensor across the feature dimension (d_model).
        """
        mean = x.mean(dim=-1, keepdim=True)  # Mean across the last dimension (d_model) 
        #keepdim is True to keep the same number of dimensions after reduction. If we don't keep it, the resulting tensor will have shape (batch_size, sequence_length) instead of (batch_size, sequence_length, 1)
        std = x.std(dim=-1, keepdim=True)    # Standard deviation across the last dimension (d_model)
        normalized_x = (x - mean) / (std + self.eps)  # Normalize
        return self.alpha * normalized_x + self.bias  # Scale and shift
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
    #(Batch_size, Sequence_length, d_model) -> (Batch_size, Sequence_length, d_ff) -> (Batch_size, Sequence_length, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        #(Batch, h, seq_len, d_K) -> (Batch, h, seq_len , seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q,k,v, mask):
        query = self.w_q(q) #(Batch, Seq_len, d_model) -> # (Batch, Seq_len, d_model)
        key = self.w_k(k) #(Batch, Seq_len, d_model) -> # (Batch, Seq_len, d_model)
        value = self.w_v(v) #(Batch, Seq_len, d_model) -> # (Batch, Seq_len, d_model)

        #(Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).tranpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h, d_k) -> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k)

        # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        x is the input tensor of shape (batch_size, sequence_length, d_model)
        sublayer is a function that takes x as input and returns a tensor of the same shape as x.
        We apply the sublayer to the normalized input, apply dropout, and then add the original
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)

    def forward(self, x, src_mask): 
        """
        we use the mask in the encoder block because we might have padding tokens in the input sequence 
        and we don't want the model to attend to those padding tokens.
        """

        x = self.residual1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers   
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self,self_attention: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()

        self.self_attention_block = self.self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connections[0](x, lambda : self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_ouput, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, encoder_ouput, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        #(Batch, Seq_Len, d_model) --> (Batch, Seq_len, Vocab_Size)
        return torch.log_softmax(self.proj(x), dim = -1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_emb: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    #Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block)
        encoder_blocks.append(encoder_block)


    # Create the decoder blocks
    decoder_blocks = []
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)

    #Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #Initialize the parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer




