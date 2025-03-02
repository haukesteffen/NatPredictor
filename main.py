import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping

from utils.data import DataConfig, create_dataloaders
from utils.model import RNN_Nationality_Predictor
from utils.config import load_config
from utils.lightning_model import LightningModelWrapper


def main():
    # Initialize MLflow logger
    mlflow_logger = MLFlowLogger(experiment_name='Nationality Predictor', log_model=True)
    
    # Load and log parameter config and data config
    config = load_config("config.yaml", mlflow_logger)
    data_config = DataConfig.from_files(
        maximum_name_length=config.parameters.maximum_name_length,
        vocab_path=config.data_parameters.vocab_path,
        country_codes_path=config.data_parameters.country_codes_path,
        target_class=config.data_parameters.target_class
    )
    print(f'Vocabulary size: {len(data_config.vocabulary)}')
    print(f'Number of country codes: {len(data_config.country_codes)}')
    print(f'Number of target classes: {len(set(data_config.country_mapping.values()))}')
    
    # handle device
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    torch.set_float32_matmul_precision('medium')

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        config=data_config,
        train_path=config.data_parameters.train_path,
        val_path=config.data_parameters.val_path,
        test_path=config.data_parameters.test_path,
        batch_size=config.parameters.batch_size
    )

    # instantiate pytorch model
    model = RNN_Nationality_Predictor(
        input_size=len(data_config.vocabulary)+1,
        output_size=len(set(data_config.country_mapping.values()))+1,
        architecture=config.hyperparameters.architecture,
        embedding_dim=config.hyperparameters.embedding_dim,
        hidden_size=config.hyperparameters.hidden_size,
        num_rnn_layers=config.hyperparameters.num_rnn_layers,
        dropout=config.hyperparameters.dropout
    )

    # wrap model in a lightning wrapper
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