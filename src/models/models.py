from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import torchmetrics

from .model_bases import STR2BASE, STR2OPTIM
from .data import CroplandClassificationDataset, FloodClassificationDataset

class CroplandMapper(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        pl.seed_everything(2022)
        self.save_hyperparameters(hparams)

        """Dataset initialization"""
        self.data_folder = Path(hparams.data_folder)
        dataset = self.get_dataset(subset = 'train')
        self.input_size = dataset.num_input_features
        self.num_outputs = dataset.num_output_classes
        self.normalizing_dict = dataset.normalizing_dict

        """Model structure"""
        self.base = STR2BASE[hparams.model_base](
            input_size = self.input_size, hparams = self.hparams
        )

        self.batchnorm = nn.BatchNorm1d(self.hparams.hidden_size)

        mlp_layers: list[nn.Module] = []
        for i in range(hparams.num_classification_layers):
            mlp_layers.extend([
                nn.Linear(
                    in_features = hparams.hidden_size if i == 0 else hparams.classifier_hidden_size,
                    out_features = self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size
                ),
                nn.BatchNorm1d(self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size),
            ])
        
        self.classifier = nn.Sequential(*mlp_layers)

        self.loss_function: Callable = F.binary_cross_entropy

        """Interpretable metrics for evaluation"""
        self.val_acc = torchmetrics.Accuracy(threshold=self.hparams.crop_probability_threshold)
        self.val_auc = torchmetrics.AUROC()
        self.val_precision = torchmetrics.Precision(threshold=self.hparams.crop_probability_threshold)
        self.val_recall = torchmetrics.Recall(threshold=self.hparams.crop_probability_threshold)
        self.val_f1 = torchmetrics.F1Score(threshold=self.hparams.crop_probability_threshold)

        self.test_acc = torchmetrics.Accuracy(threshold=self.hparams.crop_probability_threshold)
        self.test_auc = torchmetrics.AUROC()
        self.test_precision = torchmetrics.Precision(threshold=self.hparams.crop_probability_threshold)
        self.test_recall = torchmetrics.Recall(threshold=self.hparams.crop_probability_threshold)
        self.test_f1 = torchmetrics.F1Score(threshold=self.hparams.crop_probability_threshold)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out = self.classifier(self.batchnorm(self.base(x)))

        if self.num_outputs == 1:
            out = torch.sigmoid(out)

        return out

    def get_dataset(self, subset: str) -> CroplandClassificationDataset:
        return CroplandClassificationDataset(
            data_folder = self.data_folder,
            subset = subset
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(subset='train'),
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(subset='validation'),
            batch_size=self.hparams.batch_size,
            num_workers=0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(subset='test'),
            batch_size=self.hparams.batch_size,
            num_workers=0
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = STR2OPTIM[self.hparams.optimizer](
            self.parameters(),
            lr = self.hparams.learning_rate,
            weight_decay = self.hparams.weight_decay
        )

        if isinstance(optimizer, torch.optim.SGD):
            for group in optimizer.param_groups:
                group['momentum'] = self.hparams.momentum
        
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.split_preds_and_get_loss(
            batch, loss_label='train_loss', log_loss=True
        )
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.split_preds_and_get_loss(
            batch, loss_label='validation_loss', log_loss=False
        )
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.split_preds_and_get_loss(
            batch, loss_label='test_loss', log_loss=False
        )
    
    def validation_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        metrics = self.get_interpretable_metrics(outputs, 'validation')

        self.log('validation_loss', avg_loss, prog_bar=True)
        self.log_dict(metrics)

    def test_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        metrics = self.get_interpretable_metrics(outputs, 'test')

        self.log('test_loss', avg_loss)
        self.log_dict(metrics)
    
    def split_preds_and_get_loss(
        self, 
        batch: torch.Tensor,
        loss_label: str, 
        log_loss: bool
    ) -> torch.Tensor:
        data, crop_labels = batch
        data, crop_labels = data.float(), crop_labels.float()
     
        """Get model outputs"""
        crop_present_y_tilde = self.forward(data)
        crop_present_y_tilde = crop_present_y_tilde.squeeze()

        """Calculate loss"""
        loss = self.loss_function(crop_present_y_tilde, crop_labels)

        output_dict: dict[str, torch.Tensor] = {
            'loss': loss,
            'crop_present_probs': crop_present_y_tilde,
            'crop_labels': crop_labels
        }

        if log_loss:
            self.log(loss_label, loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        
        return output_dict
    
    def get_interpretable_metrics(self, outputs: dict[str, torch.Tensor], prefix: str) -> dict:
        """Helper function to return more interpretable metrics from predictions"""
        output_dict = {}

        output_dict.update(
            self.single_output_metrics(
                torch.cat([x['crop_present_probs'] for x in outputs]),
                torch.cat([x['crop_labels'] for x in outputs]).int(),
                f'{prefix}'
            )
        )

        return output_dict

    def single_output_metrics(self, probs: torch.Tensor, labels: torch.Tensor, prefix: str) -> dict:
        if len(probs) == 0:
            return {}

        if prefix == 'validation':
            self.val_acc(probs, labels)
            self.val_auc(probs, labels)
            self.val_precision(probs, labels)
            self.val_recall(probs, labels)
            self.val_f1(probs, labels)

            metrics: dict[str, torchmetrics.Metric] = {
                f'{prefix}_accuracy': self.val_acc,
                f'{prefix}_auc': self.val_auc,
                f'{prefix}_precision': self.val_precision,
                f'{prefix}_recall': self.val_recall,
                f'{prefix}_f1': self.val_f1
            }
        
        elif prefix == 'test':
            self.test_acc(probs, labels)
            self.test_auc(probs, labels)
            self.test_precision(probs, labels)
            self.test_recall(probs, labels)
            self.test_f1(probs, labels)

            metrics: dict[str, torchmetrics.Metric] = {
                f'{prefix}_accuracy': self.test_acc,
                f'{prefix}_auc': self.test_auc,
                f'{prefix}_precision': self.test_precision,
                f'{prefix}_recall': self.test_recall,
                f'{prefix}_f1': self.test_f1
            }

        return metrics
        
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser_args: dict[str, tuple[type, any]] = {
            '--data_folder': (str, str(Path('../data').absolute())),
            '--model_base': (str, 'lstm'),
            '--hidden_size': (int, 128),
            '--classifier_hidden_size': (int, 256),
            '--optimizer': (str, 'adam'), 
            '--learning_rate': (float, 0.001),
            '--weight_decay': (float, 0.0001),
            '--batch_size': (int, 64),
            '--num_classification_layers': (int, 2),
            '--crop_probability_threshold': (float, 0.5)
        }

        for k, v in parser_args.items():
            parser.add_argument(k, type=v[0], default=v[1])

        parser.add_argument('--auto_lr', dest='auto_lr', action='store_true')
        parser.add_argument('--momentum', nargs='?', type=float, default=0.9)
        
        temp_args = parser.parse_known_args()[0]
        return STR2BASE[temp_args.model_base].add_base_specific_arguments(parser)

class FloodMapper(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        pl.seed_everything(2022)
        self.save_hyperparameters(hparams)

        """Dataset initialization"""
        self.data_folder = Path(hparams.data_folder)
        dataset = self.get_dataset(subset = 'train')
        self.input_size = dataset.num_input_features
        self.num_outputs = dataset.num_output_classes
        self.normalizing_dict = dataset.normalizing_dict

        """Model structure"""
        self.base = STR2BASE[hparams.model_base](
            input_size = self.input_size, hparams = self.hparams
        )

        self.batchnorm = nn.BatchNorm1d(self.hparams.hidden_size)

        mlp_layers: list[nn.Module] = []
        for i in range(hparams.num_classification_layers):
            mlp_layers.extend([
                nn.Linear(
                    in_features = hparams.hidden_size if i == 0 else hparams.classifier_hidden_size,
                    out_features = self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size
                ),
                nn.BatchNorm1d(self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size),
                nn.GELU()
            ])
        
        self.present_classifier = nn.Sequential(*mlp_layers)

        if self.hparams.multi_headed:
            mlp_layers: list[nn.Module] = []
            for i in range(hparams.num_classification_layers):
                mlp_layers.extend([
                    nn.Linear(
                        in_features = hparams.hidden_size if i == 0 else hparams.classifier_hidden_size,
                        out_features = self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size
                    ),
                    nn.BatchNorm1d(self.num_outputs if i == (hparams.num_classification_layers - 1) else hparams.classifier_hidden_size),
                    nn.GELU()
                ])
            
            self.past_classifier = nn.Sequential(*mlp_layers)

        self.loss_function: Callable = F.binary_cross_entropy

        """Interpretable metrics for evaluation"""
        self.val_acc = torchmetrics.Accuracy(threshold=self.hparams.flood_probability_threshold)
        self.val_auc = torchmetrics.AUROC()
        self.val_precision = torchmetrics.Precision(threshold=self.hparams.flood_probability_threshold)
        self.val_recall = torchmetrics.Recall(threshold=self.hparams.flood_probability_threshold)
        self.val_f1 = torchmetrics.F1Score(threshold=self.hparams.flood_probability_threshold)

        self.test_acc = torchmetrics.Accuracy(threshold=self.hparams.flood_probability_threshold)
        self.test_auc = torchmetrics.AUROC()
        self.test_precision = torchmetrics.Precision(threshold=self.hparams.flood_probability_threshold)
        self.test_recall = torchmetrics.Recall(threshold=self.hparams.flood_probability_threshold)
        self.test_f1 = torchmetrics.F1Score(threshold=self.hparams.flood_probability_threshold)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        present_out = self.present_classifier(self.batchnorm(self.base(x)))

        if self.num_outputs == 1:
            present_out = torch.sigmoid(present_out)
        
        if self.hparams.multi_headed:
            past_out = self.past_classifier(self.batchnorm(self.base(x)))
            if self.num_outputs == 1:
                past_out = torch.sigmoid(past_out)
            return present_out, past_out     
        else:      
            return present_out

    def get_dataset(self, subset: str) -> FloodClassificationDataset:
        return FloodClassificationDataset(
            data_folder = self.data_folder,
            subset = subset
        )

    def train_dataloader(self) -> DataLoader:
        """Use a weighted sampler to deal with class imbalance"""
        train_dataset = self.get_dataset(subset='train')
        
        weighted_sampler = WeightedRandomSampler(
            train_dataset.output_class_weights, 
            len(train_dataset.output_class_weights),
            replacement=False
        )

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            sampler=weighted_sampler
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(subset='validation'),
            batch_size=self.hparams.batch_size,
            num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(subset='test'),
            batch_size=self.hparams.batch_size,
            num_workers=4
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = STR2OPTIM[self.hparams.optimizer](
            self.parameters(),
            lr = self.hparams.learning_rate,
            weight_decay = self.hparams.weight_decay
        )

        if isinstance(optimizer, torch.optim.SGD):
            for group in optimizer.param_groups:
                group['momentum'] = self.hparams.momentum
        
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.split_preds_and_get_loss(
            batch, loss_label='train_loss', log_loss=True
        )
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.split_preds_and_get_loss(
            batch, loss_label='validation_loss', log_loss=False
        )
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.split_preds_and_get_loss(
            batch, loss_label='test_loss', log_loss=False
        )
    
    def validation_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        metrics = self.get_interpretable_metrics(outputs, 'validation')

        self.log('validation_loss', avg_loss, prog_bar=True)
        self.log_dict(metrics)

    def test_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        metrics = self.get_interpretable_metrics(outputs, 'test')

        self.log('test_loss', avg_loss)
        self.log_dict(metrics)

    def split_preds_and_get_loss(
        self, 
        batch: torch.Tensor,
        loss_label: str, 
        log_loss: bool
    ) -> torch.Tensor:
        data, flood_present_labels, flood_past_labels = batch
        data = data.float()
        flood_present_labels = flood_present_labels.float()
        flood_past_labels = flood_past_labels.float()

        loss = 0
        if self.hparams.multi_headed:
            """Get model outputs"""
            flood_present_y_tilde, flood_past_y_tilde = self.forward(data)
            flood_present_y_tilde, flood_past_y_tilde = flood_present_y_tilde.squeeze(), flood_past_y_tilde.squeeze()

            """Calculate loss"""
            present_loss = self.loss_function(flood_present_y_tilde, flood_present_labels)
            prior_loss = self.loss_function(flood_past_y_tilde, flood_past_labels)
            loss += (present_loss + self.hparams.alpha * prior_loss)

            output_dict: dict[str, torch.Tensor] = {
                'flood_present_probs': flood_present_y_tilde,
                'flood_past_probs': flood_past_y_tilde,
                'flood_present_labels': flood_present_labels,
                'flood_past_labels': flood_past_labels
            }
        else:
            """Get model outputs"""
            flood_present_y_tilde = self.forward(data)
            flood_present_y_tilde = flood_present_y_tilde.squeeze()

            """Calculate loss"""
            present_loss = self.loss_function(flood_present_y_tilde, flood_present_labels)
            loss += present_loss

            output_dict: dict[str, torch.Tensor] = {
                'flood_present_probs': flood_present_y_tilde,
                'flood_present_labels': flood_present_labels
            }

        output_dict['loss'] = loss
        if log_loss:
            self.log(loss_label, loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        
        return output_dict
    
    def get_interpretable_metrics(self, outputs: dict[str, torch.Tensor], prefix: str) -> dict:
        """Helper function to return more interpretable metrics from predictions"""
        output_dict = {}

        output_dict.update(
            self.single_output_metrics(
                torch.cat([x['flood_present_probs'] for x in outputs]),
                torch.cat([x['flood_present_labels'] for x in outputs]).int(),
                f'{prefix}_flood_present'
            )
        )

        if self.hparams.multi_headed:
            output_dict.update(
            self.single_output_metrics(
                torch.cat([x['flood_past_probs'] for x in outputs]),
                torch.cat([x['flood_past_labels'] for x in outputs]).int(),
                f'{prefix}_flood_past'
                )
            )

        return output_dict

    def single_output_metrics(self, probs: torch.Tensor, labels: torch.Tensor, prefix: str) -> dict:
        if len(probs) == 0:
            return {}
        
        if 'validation' in prefix:
            self.val_acc(probs, labels)
            self.val_auc(probs, labels)
            self.val_precision(probs, labels)
            self.val_recall(probs, labels)
            self.val_f1(probs, labels)

            metrics: dict[str, torchmetrics.Metric] = {
                f'{prefix}_accuracy': self.val_acc,
                f'{prefix}_auc': self.val_auc,
                f'{prefix}_precision': self.val_precision,
                f'{prefix}_recall': self.val_recall,
                f'{prefix}_f1': self.val_f1
            }
        
        elif 'test' in prefix:
            self.test_acc(probs, labels)
            self.test_auc(probs, labels)
            self.test_precision(probs, labels)
            self.test_recall(probs, labels)
            self.test_f1(probs, labels)

            metrics: dict[str, torchmetrics.Metric] = {
                f'{prefix}_accuracy': self.test_acc,
                f'{prefix}_auc': self.test_auc,
                f'{prefix}_precision': self.test_precision,
                f'{prefix}_recall': self.test_recall,
                f'{prefix}_f1': self.test_f1
            }

        return metrics

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser_args: dict[str, tuple[type, any]] = {
            '--data_folder': (str, str(Path('../data').absolute())),
            '--model_base': (str, 'lstm'),
            '--hidden_size': (int, 128),
            '--classifier_hidden_size': (int, 256),
            '--optimizer': (str, 'adam'), 
            '--learning_rate': (float, 0.001),
            '--weight_decay': (float, 0.0001),
            '--batch_size': (int, 64),
            '--num_classification_layers': (int, 2),
            '--alpha': (float, 0.5),
            '--flood_probability_threshold': (float, 0.5)
        }

        for k, v in parser_args.items():
            parser.add_argument(k, type=v[0], default=v[1])

        parser.add_argument('--multi_headed', dest='multi_headed', action='store_true')
        parser.add_argument('--not_multi_headed', dest='multi_headed', action='store_false')
        parser.set_defaults(multi_headed=True)

        parser.add_argument('--auto_lr', dest='auto_lr', action='store_true')
        parser.add_argument('--momentum', nargs='?', type=float, default=0.9)
        
        temp_args = parser.parse_known_args()[0]
        return STR2BASE[temp_args.model_base].add_base_specific_arguments(parser)
