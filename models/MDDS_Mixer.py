import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine: x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patch, feature_dim, dropout=0.1, expansion_factor=2):
        super().__init__()
        self.norm_time = nn.BatchNorm1d(feature_dim)
        self.mlp_time = Mlp(in_features=num_patch, hidden_features=num_patch * expansion_factor, dropout=dropout)
        self.norm_feat = nn.BatchNorm1d(feature_dim)
        self.mlp_feat = Mlp(in_features=feature_dim, hidden_features=feature_dim * expansion_factor, dropout=dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.norm_time(x)
        x = self.mlp_time(x)
        x = x.transpose(1, 2)
        x = x + residual
        residual = x
        x = x.transpose(1, 2)
        x = self.norm_feat(x)
        x = x.transpose(1, 2)
        x = self.mlp_feat(x)
        x = x + residual
        return x


class CausalDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad_len = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, dilation=dilation, groups=in_ch, padding=0)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if self.pad_len > 0: x = F.pad(x, (self.pad_len, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiScaleDilatedDepthwiseConvEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, d_patch, stride=None, num_channels=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len // 2
        self.num_patch = (seq_len - patch_len) // self.stride + 1
        self.d_patch = d_patch
        conv_channels = d_patch // 3 if d_patch >= 3 else 1
        if conv_channels == 0: conv_channels = 1
        self.conv_d1 = CausalDepthwiseSeparableConv(num_channels, conv_channels, kernel_size=3, dilation=1)
        self.conv_d2 = CausalDepthwiseSeparableConv(num_channels, conv_channels, kernel_size=3, dilation=2)
        self.conv_d4 = CausalDepthwiseSeparableConv(num_channels, conv_channels, kernel_size=3, dilation=4)
        self.projection = nn.Conv1d(conv_channels * 3, d_patch, kernel_size=1)
        self.pe_global = nn.Parameter(torch.randn(1, self.num_patch, 1, d_patch))
        self.pe_local = nn.Parameter(torch.randn(1, 1, patch_len, d_patch))

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        f1 = self.conv_d1(x)
        f2 = self.conv_d2(x)
        f4 = self.conv_d4(x)
        features = self.projection(torch.cat([f1, f2, f4], dim=1)).permute(0, 2, 1)
        x = features.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)
        x = x + self.pe_global + self.pe_local
        x = x.contiguous().view(B, self.num_patch, -1)
        return x


class Model(nn.Module):
    """
    MDDS_Mixer: Multi-Scale Dilated Depthwise Conv Embedding + MLP-Mixer
    with RevIN normalization and Channel Independence.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full_model')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout
        self.e_layers = configs.e_layers
        self.d_patch = getattr(configs, 'd_patch', configs.d_model)
        self.patch_len = getattr(configs, 'patch_len', 16)

        if self.ablation_mode != 'w/o RevIN':
            self.revin = RevIN(self.enc_in, affine=True)

        num_embedding_channels = 1 if self.ablation_mode != 'w/o Channel Independence' else self.enc_in
        self.stride = self.patch_len // 2
        self.seasonal_embedding = MultiScaleDilatedDepthwiseConvEmbedding(
            seq_len=self.seq_len, patch_len=self.patch_len, d_patch=self.d_patch,
            stride=self.stride, num_channels=num_embedding_channels
        )
        self.num_patch = self.seasonal_embedding.num_patch
        self.actual_feature_dim = self.patch_len * self.d_patch

        if self.ablation_mode != 'w/o Mixer':
            self.seasonal_blocks = nn.ModuleList([
                MixerBlock(num_patch=self.num_patch, feature_dim=self.actual_feature_dim, dropout=self.dropout,
                           expansion_factor=2)
                for _ in range(self.e_layers)
            ])

        self.seasonal_head = nn.Linear(self.num_patch * self.actual_feature_dim, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc
        if self.ablation_mode != 'w/o RevIN':
            x = self.revin(x, 'norm')
        res = x
        B, L, C = res.shape
        if self.ablation_mode != 'w/o Channel Independence':
            res_for_embedding = res.permute(0, 2, 1).reshape(B * C, L, 1)
        else:
            res_for_embedding = res
        x_enc_embed = self.seasonal_embedding(res_for_embedding)
        if self.ablation_mode != 'w/o Mixer':
            for block in self.seasonal_blocks:
                x_enc_embed = block(x_enc_embed)
        x_enc_flat = x_enc_embed.reshape(B * C, -1)
        pred_seasonal = self.seasonal_head(x_enc_flat)
        pred_seasonal = pred_seasonal.reshape(B, C, self.pred_len).permute(0, 2, 1)
        output = pred_seasonal
        if self.ablation_mode != 'w/o RevIN':
            output = self.revin(output, 'denorm')
        return output
