"""TE quality predictors with a scalar endpoint output.

LSTM, PI-LSTM and Transformer add three relative within-window position
features internally. PI-LSTM uses the LSTM architecture; its physical loss is
configured by the trainer. All methods within a predictor use the same model.
"""
import copy
import math
import numbers

import numpy as np
import torch
from torch import nn


SENSOR_DIM = 23
WINDOW_LENGTH = 20

DEFAULT_CONFIGS = {
    'lstm': {'hidden_dim': 64, 'num_layers': 2, 'dropout': .2},
    'pilstm': {'hidden_dim': 64, 'num_layers': 2, 'dropout': .2},
    'cnn': {'dropout': .6},
    'transformer': {'d_model': 12, 'nhead': 4, 'num_layers': 2, 'dropout': .1},
}


def _positions():
    # Match the original NumPy float64 feature calculation followed by its
    # torch.FloatTensor cast; broadcasting avoids storing one copy per sample.
    index = np.arange(WINDOW_LENGTH)
    values = np.stack((index / WINDOW_LENGTH,
                       np.sin(2 * np.pi * index / WINDOW_LENGTH),
                       np.cos(2 * np.pi * index / WINDOW_LENGTH)), axis=-1)
    return torch.from_numpy(values.astype(np.float32)).unsqueeze(0)


def _check_window(x):
    if not torch.is_tensor(x) or x.ndim != 3 or tuple(x.shape[1:]) != (20, 23):
        raise ValueError('Expected a sensor window with shape (N,20,23)')
    if x.shape[0] < 1 or not x.is_floating_point():
        raise ValueError('Sensor windows must be nonempty floating tensors')


class EndpointLSTM(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, dropout=.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.register_buffer('relative_position_features', _positions())
        self.lstm = nn.LSTM(
            SENSOR_DIM + 3, hidden_dim, num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        _check_window(x)
        x = torch.cat((x, self.relative_position_features.expand(len(x), -1, -1)), dim=-1)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(self.dropout(context_vector))


class EndpointCNN(nn.Module):
    def __init__(self, dropout=.6):
        super().__init__()
        self.input_channels = SENSOR_DIM
        self.seq_length = WINDOW_LENGTH
        self.conv1 = nn.Sequential(
            nn.Conv1d(SENSOR_DIM, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        _check_window(x)
        x = self.conv3(self.conv2(self.conv1(x.transpose(1, 2))))
        return self.fc(self.global_avg_pool(x).squeeze(-1))


class EndpointTransformer(nn.Module):
    def __init__(self, d_model=12, nhead=4, num_layers=2, dropout=.1):
        super().__init__()
        self.register_buffer('relative_position_features', _positions())
        self.input_projection = nn.Linear(SENSOR_DIM + 3, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, WINDOW_LENGTH, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        _check_window(x)
        x = torch.cat((x, self.relative_position_features.expand(len(x), -1, -1)), dim=-1)
        x = self.input_projection(x) + self.pos_encoder
        output = self.transformer_encoder(x)
        return self.output_layer(output[:, -1, :])


def build_model(kind, input_dim=23, config=None):
    """Build one shared architecture for Raw and augmented data methods.

    Architecture parameters are independent of the generator and physical loss.
    """
    kind = str(kind).lower()
    if kind not in DEFAULT_CONFIGS:
        raise ValueError('Unknown predictor: ' + kind)
    if input_dim != SENSOR_DIM:
        raise ValueError('This experiment requires exactly 23 measured inputs')
    resolved = copy.deepcopy(DEFAULT_CONFIGS[kind])
    supplied = dict(config or {})
    if set(supplied) - set(resolved):
        raise ValueError('Unknown architecture parameter(s): ' + str(sorted(set(supplied) - set(resolved))))
    resolved.update(supplied)
    for name, value in resolved.items():
        if name == 'dropout':
            if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError('dropout must be finite and in [0,1)')
        elif not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if kind == 'transformer' and resolved['d_model'] % resolved['nhead']:
        raise ValueError('d_model must be divisible by nhead')
    if kind in ('lstm', 'pilstm'):
        model = EndpointLSTM(**resolved)
    elif kind == 'cnn':
        model = EndpointCNN(**resolved)
    else:
        model = EndpointTransformer(**resolved)
    model.predictor_kind = kind
    model.predictor_config = copy.deepcopy(resolved)
    return model
