import pandas as pd
import numpy as np
import country_converter as coco
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from utils.data import NameNationalityData, NameNationalityDataStream
from utils.model import RNN_Nationality_Predictor, Transformer_Nationality_Predictor
from utils.config import load_config
from utils.lightning_model import LightningModelWrapper
from sklearn.metrics import roc_auc_score, top_k_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import yaml



def main():
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    torch.set_float32_matmul_precision('medium')

    # read config file
    config = load_config("config.yaml")

    # read country codes
    with open('./data/.country_codes', 'r') as f:
        COUNTRY_CODES: list = f.read().splitlines()
    print(f'Country codes: {", ".join(COUNTRY_CODES)}')

    #read vocabulary (all unique characters used in the dataset)
    with open('./data/.vocabulary', 'r') as f:
        VOCABULARY: str = f.read()
    print(f'Vocabulary: {VOCABULARY}')

    # generate country code mappings
    target_class: str = 'UNregion' # see country_converter documentation on PyPI for available classes
    COUNTRY_MAPPING: dict = {cc: coco.convert(names=cc, to=target_class) for cc in COUNTRY_CODES} 
    print(f'Target classes: {", ".join(list(set(COUNTRY_MAPPING.values())))}')



    train_data = NameNationalityDataStream(
        data_file='./data/train.csv',
        chunksize=config.parameters.batch_size,
        maximum_name_length=config.parameters.maximum_name_length,
        vocabulary=VOCABULARY,
        country_codes=COUNTRY_CODES,
        country_mapping=COUNTRY_MAPPING
    )
    train_dataloader = DataLoader(train_data, batch_size=config.parameters.batch_size, num_workers=8)

    val_data = NameNationalityData(
        data_file='./data/val.csv',
        maximum_name_length=config.parameters.maximum_name_length,
        vocabulary=VOCABULARY,
        country_codes=COUNTRY_CODES,
        country_mapping=COUNTRY_MAPPING
    )
    val_dataloader = DataLoader(val_data, batch_size=config.parameters.batch_size, num_workers=8, persistent_workers=True)

    test_data = NameNationalityData(
        data_file='./data/test.csv',
        maximum_name_length=config.parameters.maximum_name_length,
        vocabulary=VOCABULARY,
        country_codes=COUNTRY_CODES,
        country_mapping=COUNTRY_MAPPING
    )
    test_dataloader = DataLoader(test_data, batch_size=config.parameters.batch_size, num_workers=8, persistent_workers=True)

    mlflow_logger = MLFlowLogger(experiment_name='Nationality Predictor', log_model=True)

    # log training parameters
    params = {
        "max_epochs": config.parameters.max_epochs,
        "batch_size": config.parameters.batch_size,
    }
    mlflow_logger.log_hyperparams(params)

    model = RNN_Nationality_Predictor(
        input_size=len(VOCABULARY)+1,
        output_size=len(set(COUNTRY_MAPPING.values()))+1,
        architecture=config.hyperparameters.architecture,
        embedding_dim=config.hyperparameters.embedding_dim,
        hidden_size=config.hyperparameters.hidden_size,
        num_rnn_layers=config.hyperparameters.num_rnn_layers,
        dropout=config.hyperparameters.dropout
    )
    lightning_model = LightningModelWrapper(
        mlflow_logger=mlflow_logger,
        model=model,
        criterion=F.cross_entropy,
        warmup_steps=config.lr_scheduler_parameters.warmup_steps,
        cosine_steps=config.lr_scheduler_parameters.cosine_steps,
        min_lr=config.lr_scheduler_parameters.min_lr,
        max_lr=config.lr_scheduler_parameters.max_lr
    )

    # register callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping('val_loss', patience=config.parameters.early_stopping_patience) # patience counts checks, not steps, so changing val_check_interval in Trainer instantiation changes this behaviour
    callbacks = [lr_monitor, early_stopping]
    mlflow_logger.log_hyperparams({'callbacks': ', '.join([callback.__class__.__name__ for callback in callbacks])})

    # instantiate trainer
    trainer = L.Trainer(
        max_epochs=config.parameters.max_epochs,
        limit_val_batches=len(val_dataloader),
        val_check_interval=config.parameters.n_eval,
        log_every_n_steps=config.parameters.n_eval,
        logger=mlflow_logger,
        callbacks=callbacks,
        gradient_clip_val=config.parameters.gradient_clipping_val
    )

    # fit model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # test model
    trainer.test(
        lightning_model,
        test_dataloader,
    )

if __name__ == "__main__":
    main()