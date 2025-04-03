import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from utils.config import load_config
from utils.metadata import initialize_metadata
from utils.encoder import NationalityEncoder
from utils.transforms import NationalityTransform
from utils.data import create_dataloaders
from utils.model import RNN_Nationality_Predictor
from utils.lightning_model import LightningModelWrapper


def main():
    # initialize MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name='Nationality Predictor',
        tracking_uri='http://localhost:5001',
        log_model=True
    )
    
    # load and log config
    config = load_config(
        path="config.yaml",
        mlflow_logger=mlflow_logger
        )

    # initialize metadata
    vocabulary, country_codes, mappings = initialize_metadata(
        config=config,
        mlflow_logger=mlflow_logger
    )

    # initialize encoder and transform
    encoder = NationalityEncoder(
        mappings=mappings,
        maximum_name_length=config.parameters.maximum_name_length
    )
    transform = NationalityTransform(encoder=encoder)
    
    # handle device
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    torch.set_float32_matmul_precision('medium')

    # create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        transform=transform,
        train_path=config.data_parameters.train_path,
        val_path=config.data_parameters.val_path,
        test_path=config.data_parameters.test_path,
        batch_size=config.parameters.batch_size
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
    lightning_model = LightningModelWrapper(
        mlflow_logger=mlflow_logger,
        model=model,
        criterion=F.cross_entropy,
        warmup_steps=config.lr_scheduler_parameters.warmup_steps,
        cosine_steps=config.lr_scheduler_parameters.cosine_steps,
        min_lr=config.lr_scheduler_parameters.min_lr,
        max_lr=config.lr_scheduler_parameters.max_lr
    )
    lightning_model = torch.compile(lightning_model)

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
        gradient_clip_val=config.parameters.gradient_clipping_val,
        precision="bf16-mixed"
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