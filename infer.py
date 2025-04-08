import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import MLFlowLogger
from typing import List
from utils.config import load_config
from utils.metadata import initialize_metadata
from utils.encoder import NationalityEncoder
from utils.lightning_model import LightningModelWrapper
from utils.model import RNN_Nationality_Predictor

@torch.no_grad()
def predict_nationalities(
    names_df: List[str],
    model: torch.nn.Module,
    encoder: NationalityEncoder,
    device: torch.device
) -> List[str]:
    """Predict nationalities using the same encoder."""
    # Encode names
    names = names_df['name'].to_list()
    name_tensor, lengths = encoder.encode_name(names)
    
    # Move to device
    name_tensor = name_tensor.to(device)
    lengths = lengths.to(device)
    
    # Get predictions
    logits = model(name_tensor, lengths)

    # Convert to dataframe
    df = pd.DataFrame(F.softmax(logits.cpu(), dim=1).numpy()).rename(columns=encoder.index_to_class)
    df['name'] = names
    df = df.set_index('name')
    df = df.drop(columns=[0])
    return df

device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# initialize MLflow logger
mlflow_logger = MLFlowLogger(experiment_name='Nationality Predictor', log_model=True)

# load config
config = load_config(
    path="config.yaml",
    mlflow_logger=None
)

# initialize metadata
vocabulary, country_codes, mappings = initialize_metadata(
    config=config,
    mlflow_logger=None
)

# initialize encoder and transform
encoder = NationalityEncoder(
    mappings=mappings,
    maximum_name_length=config.parameters.maximum_name_length
)

# instantiate pytorch model
model = RNN_Nationality_Predictor(
    input_size=len(vocabulary)+1, # +1 for padding
    output_size=len(mappings['country']['class_to_index'])+1, # +1 for padding
    architecture=config.hyperparameters.architecture,
    embedding_dim=config.hyperparameters.embedding_dim,
    hidden_size=config.hyperparameters.hidden_size,
    num_rnn_layers=config.hyperparameters.num_rnn_layers,
    dropout=config.hyperparameters.dropout
)

# wrap model in a lightning wrapper
lightning_model = LightningModelWrapper.load_from_checkpoint(
    'best-model.ckpt',
    map_location=device,
    mlflow_logger=mlflow_logger,
    model=model,
    criterion=F.cross_entropy,
    warmup_steps=config.lr_scheduler_parameters.warmup_steps,
    cosine_steps=config.lr_scheduler_parameters.cosine_steps,
    min_lr=config.lr_scheduler_parameters.min_lr,
    max_lr=config.lr_scheduler_parameters.max_lr
)

lightning_model.eval()
torch_model = lightning_model.model

df = pd.read_csv('test.csv')
df_out = predict_nationalities(
    names_df=df,
    model=lightning_model.model,
    encoder=encoder,
    device=device
).to_csv('predictions.csv', index=True)