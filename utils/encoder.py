import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

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
        self.country_mapping = country_mappings['country_mapping']
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