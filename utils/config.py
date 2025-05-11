from pathlib import Path
from pydantic import BaseModel
import yaml
from typing import Optional, Union
from lightning.pytorch.loggers import MLFlowLogger

class Parameters(BaseModel):
    """General training parameters.
    
    Attributes:
        batch_size (int): Batch size for training
        max_epochs (int): Maximum number of training epochs
        n_eval (int): Frequency of validation checks (in steps)
        early_stopping_patience (int): Patience for early stopping (in checks)
        gradient_clipping_val (float): Gradient clipping value
        maximum_name_length (int): Maximum length of a name
    """
    batch_size: int
    max_epochs: int
    n_eval: int
    early_stopping_patience: int
    gradient_clipping_val: float
    maximum_name_length: int

class Hyperparameters(BaseModel):
    """Model architecture hyperparameters.
    
    Attributes:
        architecture (str): RNN architecture type ('RNN', 'GRU', or 'LSTM')
        embedding_dim (int): Dimension of the character embedding space
        hidden_size (int): Number of features in the hidden state
        num_rnn_layers (int): Number of stacked RNN layers
        dropout (float): Dropout probability between layers
    """
    architecture: str
    embedding_dim: int
    hidden_size: int
    num_rnn_layers: int
    dropout: float

class LrSchedulerParameters(BaseModel):
    """Learning rate scheduler configuration.
    
    Attributes:
        warmup_steps (int): Number of steps for linear warmup
        cosine_steps (int): Number of steps for cosine annealing
        min_lr (float): Minimum learning rate after decay
        max_lr (float): Maximum learning rate at peak of warmup
    """
    warmup_steps: int
    cosine_steps: int
    min_lr: float
    max_lr: float

class DataParameters(BaseModel):
    """Data file paths.
    
    Attributes:
        train_path (Path): Path to training data
        val_path (Path): Path to validation data
        test_path (Path): Path to test data
    """
    train_path: Path
    val_path: Path
    test_path: Path

    class Config:
        arbitrary_types_allowed = True

class Config(BaseModel):
    """Main configuration class combining all parameter groups.
    
    Attributes:
        parameters (Parameters): General training parameters
        hyperparameters (Hyperparameters): Model architecture parameters
        lr_scheduler_parameters (LrSchedulerParameters): Learning rate scheduler parameters
        metadata_parameters (MetadataParameters): Metadata file paths and configurations
        data_parameters (DataParameters): Data file paths
    """
    parameters: Parameters
    hyperparameters: Hyperparameters
    lr_scheduler_parameters: LrSchedulerParameters
    data_parameters: DataParameters

def _convert_paths(data: dict) -> dict:
    """Recursively convert all dictionary values containing 'path' to Path objects."""
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = _convert_paths(value)
        elif isinstance(value, str) and 'path' in key.lower():
            data[key] = Path(value)
    return data

def load_config(path: Union[str, Path], mlflow_logger: Optional[MLFlowLogger] = None) -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert all 'path' strings to Path objects
    config = _convert_paths(config)
    
    # Instantiate Config class
    config = Config(**config)
    
    if mlflow_logger:
        mlflow_logger.log_hyperparams(config.model_dump())
    print(f"Config loaded from {path}.")
    return config
