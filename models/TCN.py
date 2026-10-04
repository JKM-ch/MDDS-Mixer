import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Crop(nn.Module):
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()


class TemporalCausalLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalCausalLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.crop1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.crop2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalCausalLayer(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.input_size = configs.enc_in
        self.time_feat_dim = 4
        self.time_feature_proj = nn.Linear(self.time_feat_dim, self.input_size)

        num_channels = [configs.d_model] * configs.e_layers
        kernel_size = 3
        dropout = configs.dropout

        self.tcn = TemporalConvolutionNetwork(
            self.input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.projection = nn.Linear(num_channels[-1], self.pred_len * self.c_out)

    def _build_time_covariates(self, x_mark_enc, x_enc):
        b, l, _ = x_enc.shape
        if x_mark_enc is None:
            return x_enc.new_zeros(b, l, self.time_feat_dim)

        x_mark = x_mark_enc[..., :self.time_feat_dim]
        if x_mark.size(-1) < self.time_feat_dim:
            pad = x_mark.new_zeros(b, l, self.time_feat_dim - x_mark.size(-1))
            x_mark = torch.cat([x_mark, pad], dim=-1)

        cov = x_mark.float()
        if cov.abs().max() > 1.5:
            cov[..., 0] = cov[..., 0] / 12.0
            cov[..., 1] = cov[..., 1] / 31.0
            cov[..., 2] = cov[..., 2] / 6.0
            cov[..., 3] = cov[..., 3] / 23.0
        return cov

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        time_cov = self._build_time_covariates(x_mark_enc, x_enc)
        x = x_enc + self.time_feature_proj(time_cov)

        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = y[:, :, -1]
        output = self.projection(y)
        output = output.reshape(output.shape[0], self.pred_len, self.c_out)
        return output
