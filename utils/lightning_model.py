import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torchinfo import summary
import lightning as L
import numpy as np
from sklearn.metrics import top_k_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score



# PyTorch Lightning Wrapper
class LightningModelWrapper(L.LightningModule):
    def __init__(self, mlflow_logger, model, criterion, warmup_steps, cosine_steps, min_lr, max_lr):
        super().__init__()
        self.mlflow_logger = mlflow_logger

        # log the scheduler hyperparameters
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mlflow_logger.log_hyperparams({
            'warmup_steps': self.warmup_steps,
            'cosine_steps': self.cosine_steps,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr
        })

        # log model summary and model hyperparameters
        self.model = model
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(self.model)))
        self.mlflow_logger.experiment.log_artifact(local_path="model_summary.txt", run_id=self.mlflow_logger.run_id)
        hyperparams = {
            "input_size": self.model.input_size,
            "output_size": self.model.output_size,
            "architecture": self.model.architecture,
            "embedding_dim": self.model.embedding_dim,
            "hidden_size": self.model.hidden_size,
            "num_rnn_layers": self.model.num_rnn_layers,
            "dropout": self.model.dropout
        }
        self.mlflow_logger.log_hyperparams(hyperparams)
        
        # log criterion
        self.criterion = criterion
        self.mlflow_logger.log_hyperparams({'criterion': self.criterion.__name__})

    def training_step(self, batch):
        X, y, seq_lengths = batch
        logits = self.model(X, seq_lengths)  # [batch_size, num_classes]
        y_true_int = torch.argmax(y, dim=1)  # [batch_size]
        loss = self.criterion(logits, y_true_int)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        X, y, seq_lengths = batch
        logits = self.model(X, seq_lengths)  # [batch_size, num_classes]
        y_true_int = torch.argmax(y, dim=1)  # [batch_size]
        loss = self.criterion(logits, y_true_int)
        self.log('val_loss', loss)

        # Calculate top-3 accuracy
        y_prob = F.softmax(logits, dim=1).detach().cpu().numpy()
        y_true_int = y_true_int.detach().cpu().numpy()
        top_3_accuracy = top_k_accuracy_score(y_true_int, y_prob, k=3, labels=list(range(logits.shape[1])))
        self.log('val_top_3_accuracy_score', top_3_accuracy)
        return loss
    
    def test_step(self, batch):
        X, y, seq_lengths = batch
        logits = self.model(X, seq_lengths)
        y_true_int = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_true_int)

        # Convert to probabilities and numpy arrays for sklearn metrics
        y_prob = F.softmax(logits, dim=1).detach().cpu().numpy()
        y_true_int = y_true_int.detach().cpu().numpy()
        y_pred_int = torch.argmax(logits, dim=1).detach().cpu().numpy()

        # Get unique classes actually present in the batch and all possible classes
        present_classes = np.unique(y_true_int)
        all_classes = list(range(logits.shape[1]))  # number of output classes

        try:
            # Calculate basic metrics using only present classes
            acc = accuracy_score(y_true_int, y_pred_int)
            macro_f1 = f1_score(y_true_int, y_pred_int, 
                               average='macro', 
                               zero_division=0,
                               labels=present_classes)
            macro_precision = precision_score(y_true_int, y_pred_int, 
                                            average='macro', 
                                            zero_division=0,
                                            labels=present_classes)
            macro_recall = recall_score(y_true_int, y_pred_int, 
                                      average='macro', 
                                      zero_division=0,
                                      labels=present_classes)
            bal_acc = balanced_accuracy_score(y_true_int, y_pred_int)

            # Calculate top-k accuracy using all possible classes
            top_3_accuracy = top_k_accuracy_score(y_true_int, y_prob, 
                                                k=3, 
                                                labels=all_classes)

            # Calculate average precision only if we have positive examples
            if len(present_classes) > 0:
                macro_avg_precision = average_precision_score(
                    y.detach().cpu().numpy(), 
                    y_prob,
                    average='macro'
                )
            else:
                macro_avg_precision = 0.0

            # Log all metrics with consistent prefix
            metrics = {
                'test_loss': loss,
                'test_accuracy': acc,
                'test_balanced_accuracy': bal_acc,
                'test_macro_f1': macro_f1,
                'test_macro_precision': macro_precision,
                'test_macro_recall': macro_recall,
                'test_macro_avg_precision': macro_avg_precision,
                'test_top_3_accuracy': top_3_accuracy
            }
            
            # Log all metrics
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True)

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Log at least the loss if metrics calculation fails
            self.log('test_loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Instantiate optimizer with the passed max_lr value.
        optimizer = optim.AdamW(self.parameters(), lr=self.max_lr)
        self.mlflow_logger.log_hyperparams({'optimizer': optimizer.__class__.__name__})

        # Warmup scheduler: scales LR from 1e-4×base to base LR over `warmup_steps`
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=self.warmup_steps)

        # Cosine annealing: decays LR from base to min_lr over `cosine_steps`
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=self.min_lr)

        # Constant scheduler: hold the LR at min_lr.
        constant_scheduler = ConstantLR(optimizer, factor=self.min_lr/self.max_lr, total_iters=1e10)

        # Combine schedulers using SequentialLR.
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_steps]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

