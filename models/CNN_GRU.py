import torch
import torch.nn as nn


class Model(nn.Module):
    """
    CNN_GRU: 1D CNN for feature extraction followed by GRU for temporal modeling.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        # 1D CNN for feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(configs.enc_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # GRU
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )

        self.projection = nn.Linear(configs.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, L, C] -> Permute for Conv1d -> [B, C, L]
        x = x_enc.permute(0, 2, 1)
        x = self.conv(x)
        # x: [B, 64, L/2] -> Permute for GRU -> [B, L/2, 64]
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last hidden state
        out = self.projection(out)
        return out.reshape(out.shape[0], self.pred_len, self.c_out)
