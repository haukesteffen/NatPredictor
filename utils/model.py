import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import country_converter as coco
from pathlib import Path
import yaml

class NationalityEncoder:
    """Central class for all encoding/decoding operations."""
    
    def __init__(self, mappings: Dict[str, Any], maximum_name_length: int):
        """Initialize encoder with mappings.
        
        Args:
            mappings: Dictionary containing character and country mappings
            maximum_name_length: Maximum length for name sequences
        """
        self.maximum_name_length = maximum_name_length
        self.padding_index = 0
        
        # Set up character mappings
        char_mappings = mappings['character']
        self.char_to_index = char_mappings['char_to_index']
        
        # Set up country mappings
        country_mappings = mappings['country']
        self.class_to_index = country_mappings['class_to_index']
        self.alpha2_to_class = country_mappings['alpha2_to_class']
        self.alpha2_to_index = country_mappings['alpha2_to_index']
        self.number_of_classes = len(self.class_to_index)
        
        # Create inverse mappings
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
    
    def encode_name(self, seq: str | List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode name(s) to index tensors."""
        if isinstance(seq, str):
            # Single name
            encoded = [self.char_to_index.get(char, self.padding_index) for char in seq]
            seq_len = len(encoded)
            padded_tensor = torch.full(
                (self.maximum_name_length,), 
                self.padding_index, 
                dtype=torch.int32
            )
            max_len = min(seq_len, self.maximum_name_length)
            padded_tensor[:max_len] = torch.tensor(encoded[:max_len], dtype=torch.int32)
            return padded_tensor, torch.tensor(max_len, dtype=torch.int32)
        else:
            # Batch of names
            encoded_input = []
            for s in seq:
                encoded_input.append([
                    self.char_to_index.get(char, self.padding_index) 
                    for char in s
                ])
            
            sequence_lengths = torch.tensor([
                min(len(encoding), self.maximum_name_length) 
                for encoding in encoded_input
            ], dtype=torch.int32)
            
            batch_size = len(encoded_input)
            padded_tensors = torch.full(
                (batch_size, self.maximum_name_length),
                self.padding_index,
                dtype=torch.int32
            )
            
            for i, encoding in enumerate(encoded_input):
                seq_len = len(encoding)
                max_len = min(seq_len, self.maximum_name_length)
                padded_tensors[i, :max_len] = torch.tensor(
                    encoding[:max_len], 
                    dtype=torch.int32
                )
            
            return padded_tensors, sequence_lengths
    
    def decode_name(self, seq_tensor: torch.Tensor) -> List[str]:
        """Decode index tensor(s) to name(s)."""
        if seq_tensor.dim() == 1:
            return [''.join([
                self.index_to_char.get(int(idx), '') 
                for idx in seq_tensor
            ])]
        elif seq_tensor.dim() == 2:
            return [''.join([
                self.index_to_char.get(int(idx), '') 
                for idx in row
            ]) for row in seq_tensor]
        else:
            raise ValueError("seq_tensor must be 1D or 2D")
    
    def encode_country(self, country_input: str | List[str]) -> torch.Tensor:
        """Encode country code(s) to one-hot tensors."""
        if isinstance(country_input, str):
            encoded_output = self.alpha2_to_index.get(country_input, self.padding_index)
        else:
            encoded_output = [
                self.alpha2_to_index.get(c, self.padding_index) 
                for c in country_input
            ]
            
        index_tensors = torch.tensor(encoded_output, dtype=torch.int64)
        return F.one_hot(
            index_tensors, 
            num_classes=self.number_of_classes+1
        ).to(torch.float32)
    
    def decode_country(self, logits: torch.Tensor) -> List[str]:
        """Convert model output logits to region labels."""
        predictions = torch.argmax(logits, dim=1)
        return [
            self.index_to_class.get(p.item(), "Unknown") 
            for p in predictions
        ]

    @classmethod
    def create_from_data(cls, train_path: Path, target_class: str, maximum_name_length: int) -> 'NationalityEncoder':
        """Create encoder from training data."""
        vocabulary, country_codes = cls._create_vocabulary_and_codes(train_path)
        mappings = cls._create_mappings(vocabulary, country_codes, target_class)
        return cls(mappings, maximum_name_length)

    @staticmethod
    def _create_vocabulary_and_codes(train_path: Path) -> Tuple[List[str], List[str]]:
        """Create vocabulary and country codes from training data."""
        vocabulary = set()
        country_codes = set()
        chunksize = 100_000

        # Process CSV in chunks to handle large files
        for chunk in pd.read_csv(train_path, chunksize=chunksize):
            # Add new characters from names
            for name in chunk['name']:
                vocabulary.update(str(name))

            # Add new country codes
            country_codes.update(chunk['alpha2'].unique())

        # Sort vocabulary and country codes
        sorted_vocab = sorted(vocabulary)
        sorted_codes = sorted(country_codes)

        return sorted_vocab, sorted_codes

    @staticmethod
    def _create_mappings(vocabulary: List[str], country_codes: List[str], target_class: str) -> Dict[str, Any]:
        """Create mappings required for the model."""
        # Create character mappings (with 0 reserved for padding)
        char_to_index = {char: idx for idx, char in enumerate(sorted(vocabulary), 1)}

        # Create country mappings (with 0 reserved for unknown)
        alpha2_to_class = {cc: coco.convert(names=cc, to=target_class)
                          for cc in country_codes}
        output_classes = sorted(set(alpha2_to_class.values()))
        class_to_index = {c: i for i, c in enumerate(output_classes, 1)}
        alpha2_to_index = {
            alpha2: class_to_index[alpha2_to_class[alpha2]]
            for alpha2 in country_codes
        }

        # Save mappings
        mappings = {
            'character': {
                'char_to_index': char_to_index,
            },
            'country': {
                'class_to_index': class_to_index,
                'alpha2_to_class': alpha2_to_class,
                'alpha2_to_index': alpha2_to_index
            }
        }
        return mappings

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

class NationalityPredictor(nn.Module):
    """Wrapper class that combines encoder and model for inference."""
    
    def __init__(
        self,
        model: RNN_Nationality_Predictor,
        encoder: Optional['NationalityEncoder'] = None,
        config: Optional[dict] = None,
        device: torch.device = None
    ):
        """
        Args:
            model: Trained RNN model
            encoder: Initialized NationalityEncoder (optional)
            config: Configuration dictionary (required if encoder is None)
            device: Device to run inference on
        """
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        if encoder is not None:
            self.encoder = encoder
        else:
            if config is None:
                raise ValueError("If encoder is not provided, config must be given.")
            train_path = config["data_parameters"]["train_path"]
            target_class = config["metadata_parameters"]["target_class"]
            maximum_name_length = config["parameters"]["maximum_name_length"]
            self.encoder = NationalityEncoder.create_from_data(
                train_path=train_path,
                target_class=target_class,
                maximum_name_length=maximum_name_length
            )
    
    def forward(self, names: List[str]) -> List[str]:
        """Forward pass for inference.
        
        Args:
            names: List of names to predict nationalities for
            
        Returns:
            List of predicted region labels
        """
        # Encode names
        name_tensor, lengths = self.encoder.encode_name(names)
        
        # Move to device
        name_tensor = name_tensor.to(self.device)
        lengths = lengths.to(self.device)
        
        # Get predictions
        logits = self.model(name_tensor, lengths)
        return self.encoder.decode_country(logits)
    
    def predict_proba(self, names: List[str]) -> torch.Tensor:
        """Get probability distribution over classes.
        
        Args:
            names: List of names to predict probabilities for
            
        Returns:
            Tensor of shape (batch_size, num_classes) with probabilities
        """
        # Encode names
        name_tensor, lengths = self.encoder.encode_name(names)
        
        # Move to device
        name_tensor = name_tensor.to(self.device)
        lengths = lengths.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(name_tensor, lengths)
            return F.softmax(logits, dim=1)
    
    def state_dict(self) -> Dict[str, Any]:
        """Save both model weights and metadata."""
        state_dict = {
            'model': self.model.state_dict(),
            'metadata': {
                'mappings': {
                    'character': {
                        'char_to_index': self.encoder.char_to_index,
                    },
                    'country': {
                        'class_to_index': self.encoder.class_to_index,
                        'alpha2_to_class': self.encoder.alpha2_to_class,
                        'alpha2_to_index': self.encoder.alpha2_to_index
                    }
                },
                'maximum_name_length': self.encoder.maximum_name_length,
                'model_config': {
                    'architecture': self.model.architecture,
                    'embedding_dim': self.model.embedding_dim,
                    'hidden_size': self.model.hidden_size,
                    'num_rnn_layers': self.model.num_rnn_layers,
                    'dropout': self.model.dropout
                }
            }
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load both model weights and metadata."""
        # Load model weights
        self.model.load_state_dict(state_dict['model'])
        
        # Load metadata into encoder
        metadata = state_dict['metadata']
        self.encoder = NationalityEncoder(
            mappings=metadata['mappings'],
            maximum_name_length=metadata['maximum_name_length']
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: torch.device = None) -> 'NationalityPredictor':
        """Create predictor from pretrained checkpoint containing both weights and metadata."""
        # Load full state dict
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        metadata = state_dict['metadata']
        
        # Create encoder
        encoder = NationalityEncoder(
            mappings=metadata['mappings'],
            maximum_name_length=metadata['maximum_name_length']
        )
        
        # Create model with config from metadata
        config = metadata['model_config']
        model = RNN_Nationality_Predictor(
            input_size=len(metadata['mappings']['character']['char_to_index'])+1,
            output_size=len(metadata['mappings']['country']['class_to_index'])+1,
            **config
        )
        
        # Create predictor and load state
        predictor = cls(model=model, encoder=encoder, device=device)
        predictor.load_state_dict(state_dict)
        
        return predictor