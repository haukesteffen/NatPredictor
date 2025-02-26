from pydantic import BaseModel
import yaml

class Parameters(BaseModel):
    maximum_name_length: int
    batch_size: int
    n_eval: int
    max_epochs: int
    early_stopping_patience: int
    gradient_clipping_val: float

class Hyperparameters(BaseModel):
    architecture: str
    embedding_dim: int
    hidden_size: int
    num_rnn_layers: int
    dropout: float

class LrSchedulerParameters(BaseModel):
    warmup_steps: int
    cosine_steps: int
    min_lr: float
    max_lr: float

# Top-level config model
class Config(BaseModel):
    parameters: Parameters
    hyperparameters: Hyperparameters
    lr_scheduler_parameters: LrSchedulerParameters

def load_config(path: str) -> Config:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)
