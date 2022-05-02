from argparse import ArgumentParser, Namespace
import math
from typing import Optional

import torch
from torch import nn
import pytorch_lightning as pl

class LSTM(pl.LightningModule):
    def __init__(self, input_size: int, hparams: Namespace) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)

        if (hparams.num_lstm_layers > 1):
            self.model = nn.LSTM(
                input_size = input_size,
                hidden_size = hparams.hidden_size,
                dropout = hparams.lstm_dropout,
                batch_first = True,
                num_layers = hparams.num_lstm_layers
            )
        else:
            self.model = UnrolledLSTM(
                input_size = input_size,
                hidden_size = hparams.hidden_size,
                dropout = hparams.lstm_dropout,
                batch_first = True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.model(x)
        return hidden[-1, :, :]

    @staticmethod
    def add_base_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents = [parent_parser], add_help=False)

        parser_args = {
            '--num_lstm_layers': (int, 1),
            '--lstm_dropout': (float, 0.2)
        }

        for key, vals in parser_args.items():
            parser.add_argument(key, type=vals[0], default=vals[1])
        
        return parser

class UnrolledLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, batch_first: bool) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.rnn = UnrolledLSTMCell(
            input_size = self.input_size, hidden_size = self.hidden_size, batch_first = self.batch_first
        )
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.parameters():
            for parameter in parameters:
                nn.init.uniform_(parameter.data, -sqrt_k, sqrt_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[tuple[torch.Tensor, torch.Tensor]]=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        sequence_length = x.size(1) if self.batch_first else x.size(0)
        batch_size = x.size(0) if self.batch_first else x.size(1)

        if state is None:
            hidden, cell = (
                torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size)
            )
        else:
            hidden, cell = state

        outputs = []
        for i in range(sequence_length):
            input_x = x[:, i, :].unsqueeze(1)
            _, (hidden, cell) = self.rnn(input_x, (hidden, cell))
            outputs.append(hidden)
            hidden = self.dropout(hidden)
        
        return torch.stack(outputs, dim=0), (hidden, cell)

        

class UnrolledLSTMCell(nn.Module):
    """
    A single-layer LSTM in which dropout can be applied between timesteps
    """

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features = input_size + hidden_size,
                    out_features = hidden_size,
                    bias = True
                ),
                nn.Sigmoid()
            ]
        )

        self.update_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features = input_size + hidden_size,
                    out_features = hidden_size,
                    bias = True
                ),
                nn.Sigmoid()
            ]
        )

        self.update_candidates = nn.Sequential(
            *[
                nn.Linear(
                    in_features = input_size + hidden_size,
                    out_features = hidden_size,
                    bias = True
                ),
                nn.Tanh()
            ]
        )

        self.output_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features = input_size + hidden_size,
                    out_features = hidden_size,
                    bias = True
                ),
                nn.Sigmoid()
            ]
        )

        self.cell_state_activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden, cell = state

        if self.batch_first:
            hidden, cell = torch.transpose(hidden, 0, 1), torch.transpose(cell, 0, 1)
        
        
        forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
        update_state = self.update_gate(torch.cat((x, hidden), dim=-1))
        cell_candidates = self.update_candidates(torch.cat((x, hidden), dim=-1))

        updated_cell = (forget_state * cell) + (update_state * cell_candidates)

        output_state = self.output_gate(torch.cat((x, hidden), dim=-1))
        updated_hidden = output_state * self.cell_state_activation(updated_cell)

        if self.batch_first:
            updated_hidden = torch.transpose(updated_hidden, 0, 1)
            updated_cell = torch.transpose(updated_cell, 0, 1)
        
        return updated_hidden, (updated_hidden, updated_cell)