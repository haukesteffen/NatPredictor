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