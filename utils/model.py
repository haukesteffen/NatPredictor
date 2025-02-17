import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class RNN_Nationality_Predictor(nn.Module):
    """
    A PyTorch-based RNN model for predicting nationality from a name.

    This model embeds input characters and processes them using a recurrent layer,
    which can be instantiated as a vanilla RNN, GRU, or LSTM. It leverages sequence
    packing to efficiently handle variable-length inputs, and uses the final hidden state
    from the RNN to produce class logits through two dense layers.

    Parameters
    ----------
    input_size : int
        The dimension of input tensors. (batch_size, input_size)
    output_size : int
        The dimension of output tensors. (batch_size, output_size)
    architecture : str
        The type of RNN to use. Must be one of: 'RNN', 'GRU', or 'LSTM'.
    embedding_dim : int
        The dimension of the embedding space for input characters.
    hidden_size : int
        The number of features in the hidden state of the RNN.
    num_rnn_layers : int
        The number of recurrent layers (stacked) in the RNN.
    dropout : float
        Dropout probability applied between RNN layers.

    Attributes
    ----------
    embed : nn.Embedding
        The embedding layer that converts input indices to dense vectors.
    rnn : nn.Module
        The recurrent layer (RNN, GRU, or LSTM) that processes the embedded sequence.
    dense1 : nn.Linear
        A linear layer that maps the final hidden state to logits.
    dense2 : nn.Linear
        A linear layer that maps the logits from the previous dense layer to output logits corresponding
        to the target nationality classes.
    """
    def __init__(self, input_size, output_size, architecture, embedding_dim, hidden_size, num_rnn_layers, dropout):
        super().__init__()
        # hyperparameters
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.architecture: str = architecture
        self.embedding_dim: int = embedding_dim
        self.hidden_size: int = hidden_size
        self.num_rnn_layers: int = num_rnn_layers
        self.dropout: float = dropout

        # embedding layer
        self.embed = nn.Embedding(
            num_embeddings=self.input_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        # rnn layers
        if architecture == 'RNN':
            rnn_constructor = nn.RNN
        elif architecture == 'GRU':
            rnn_constructor = nn.GRU
        elif architecture == 'LSTM':
            rnn_constructor = nn.LSTM
        else:
            raise NameError("architecture must be 'RNN', 'GRU' or 'LSTM'")
        self.rnn = rnn_constructor(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # output layers
        self.dense1 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.dense2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
        )

    def forward(self, X, lengths):
        embeddings = self.embed(X)

        # Pack the padded batch
        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        if self.architecture == 'LSTM':
            _, (hidden, _) = self.rnn(packed) # output and cell state ignored
        else:
            _, hidden = self.rnn(packed) # output ignored
        logits = F.relu(self.dense1(hidden[-1]))
        logits = self.dense2(self.dropout_layer(logits))
        return logits


class Transformer_Nationality_Predictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            output_size (int): Number of output classes.
            context_size (int): Maximum sequence length (without CLS token).
            d_model (int): Embedding dimension.
            nhead (int): Number of heads in multi-head attention.
            num_layers (int): Number of TransformerDecoder layers.
            dim_feedforward (int): Dimension of the feedforward network inside Transformer layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.context_size: int = context_size
        self.d_model: int = d_model
        self.nhead: int = nhead
        self.num_layers: int = num_layers
        self.dim_feedforward: int = dim_feedforward
        self.dropout: float = dropout
        
        # Token embedding (0 is used as padding_idx)
        self.token_embedding = nn.Embedding(self.input_size, self.d_model, padding_idx=0)
        
        # Positional embedding: We reserve position 0 for the classification token.
        self.pos_embedding = nn.Embedding(self.context_size + 1, self.d_model)
        
        # Learned classification token (CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        # Define a TransformerDecoderLayer and stack them
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc_out = nn.Linear(self.d_model, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, context_size) with token indices.
        
        Returns:
            Tensor: Output logits of shape (batch_size, output_size).
        """
        batch_size, seq_len = x.size()  # seq_len should be <= context_size
        
        # 1. Embed the input tokens.
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 2. Create positional indices for the tokens (starting at 1; reserve 0 for CLS).
        pos_indices = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(pos_indices)  # (batch_size, seq_len, d_model)
        
        # Sum token and positional embeddings.
        x_emb = self.dropout_layer(token_emb + pos_emb)  # (batch_size, seq_len, d_model)
        
        # 3. Prepend the classification token to the sequence.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        # The target for the decoder is now [CLS, token1, token2, ..., token_seq_len].
        tgt = torch.cat([cls_tokens, x_emb], dim=1)  # (batch_size, 1 + seq_len, d_model)
        
        # 4. Use the token embeddings (without CLS) as memory.
        memory = x_emb  # (batch_size, seq_len, d_model)
        
        # The TransformerDecoder expects inputs of shape (seq_length, batch_size, d_model).
        tgt = tgt.transpose(0, 1)      # (1 + seq_len, batch_size, d_model)
        memory = memory.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        # 5. Pass through the TransformerDecoder.
        # (You can add a tgt_mask or memory_mask here if needed.)
        decoded = self.transformer_decoder(tgt, memory)  # (1 + seq_len, batch_size, d_model)
        
        # 6. Use the output corresponding to the CLS token (first token) for classification.
        cls_decoded = decoded[0]  # (batch_size, d_model)
        logits = self.fc_out(cls_decoded)  # (batch_size, output_size)
        
        return logits
