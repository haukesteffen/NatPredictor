from typing import Tuple
import torch
from .encoder import NationalityEncoder

class NationalityTransform:
    """Transform using the centralized encoder."""
    
    def __init__(self, encoder: NationalityEncoder):
        self.encoder = encoder
    
    def __call__(self, name: str, country_code: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform raw data into model inputs."""
        # Use the centralized encoder
        name_tensor, seq_length = self.encoder.encode_name(name)
        country_tensor = self.encoder.encode_country(country_code)
        return name_tensor, country_tensor, seq_length