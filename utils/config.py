from pydantic import BaseModel
import yaml
from typing import Optional
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

class DataParameters(BaseModel):
    """Data file paths and configurations.
    
    Attributes:
        target_class (str): Target classification type (see country_converter on PyPI)
        vocab_path (str): Path to vocabulary file
        country_codes_path (str): Path to country codes file
        train_path (str): Path to training data
        val_path (str): Path to validation data
        test_path (str): Path to test data
    """
    target_class: str
    vocab_path: str
    country_codes_path: str
    train_path: str
    val_path: str
    test_path: str

class Config(BaseModel):
    """Main configuration class combining all parameter groups.
    
    Attributes:
        parameters (Parameters): General training parameters
        hyperparameters (Hyperparameters): Model architecture parameters
        lr_scheduler_parameters (LrSchedulerParameters): Learning rate scheduler parameters
        data_parameters (DataPaths): Data file paths and configurations
    """
    parameters: Parameters
    hyperparameters: Hyperparameters
    lr_scheduler_parameters: LrSchedulerParameters
    data_parameters: DataParameters

def load_config(path: str, mlflow_logger: Optional[MLFlowLogger] = None) -> Config:
    """Load configuration from a YAML file and optionally log it to MLflow.
    
    Args:
        path (str): Path to the YAML configuration file
        mlflow_logger: Optional MLflow logger to log config as artifact
        
    Returns:
        Config: Parsed configuration object
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    
    config = Config(**data)
    
    if mlflow_logger:
        # Log config as artifact
        mlflow_logger.experiment.log_artifact(
            local_path=path,
            run_id=mlflow_logger.run_id
        )
    
    return config
