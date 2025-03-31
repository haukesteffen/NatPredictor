from pathlib import Path
from pydantic import BaseModel
import yaml
from typing import Optional, Union
from lightning.pytorch.loggers import MLFlowLogger

class Parameters(BaseModel):
    """Training and evaluation parameters.
    
    Attributes:
        maximum_name_length (int): Maximum number of characters allowed in sequence
        batch_size (int): Number of samples per training batch
        n_eval (int): Frequency of evaluation in batches
        max_epochs (int): Maximum number of training epochs
        early_stopping_patience (int): Number of evaluations without improvement before stopping
        gradient_clipping_val (float): Maximum allowed gradient norm for clipping
    """
    maximum_name_length: int
    batch_size: int
    n_eval: int
    max_epochs: int
    early_stopping_patience: int
    gradient_clipping_val: float

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

class MetadataParameters(BaseModel):
    """Metadata file paths and configurations.
    
    Attributes:
        target_class (str): Target classification type (see country_converter on PyPI)
        vocab_path (Path): Path to vocabulary file
        country_codes_path (Path): Path to country codes file
        mappings_path (Path): Path to mappings file
    """
    target_class: str
    vocab_path: Path
    country_codes_path: Path
    mappings_path: Path

    class Config:
        arbitrary_types_allowed = True

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
    metadata_parameters: MetadataParameters
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
    """Load configuration from a YAML file and optionally log it to MLflow.
    
    Args:
        path (Union[str, Path]): Path to the YAML configuration file
        mlflow_logger: Optional MLflow logger to log config as artifact
        
    Returns:
        Config: Parsed configuration object
    """
    # Load YAML config file
    path = Path(path)
    with path.open('r') as file:
        data = yaml.safe_load(file)
    
    # Convert all paths in the config and create config object
    data = _convert_paths(data)
    config = Config(**data)
    print(f"Config loaded from {path}.")
    
    if mlflow_logger:
        # Log config as artifact
        mlflow_logger.experiment.log_artifact(
            local_path=str(path),
            run_id=mlflow_logger.run_id
        )

    return config
