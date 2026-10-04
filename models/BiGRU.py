import torch
import torch.nn as nn


class Model(nn.Module):
    """
    BiGRU: Bidirectional GRU for time series forecasting.
    Uses the last hidden state from both directions for prediction.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.time_feat_dim = 4
        self.time_feature_proj = nn.Linear(self.time_feat_dim, self.enc_in)

        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=configs.dropout if self.num_layers > 1 else 0,
        )
        self.projection = nn.Linear(self.hidden_size * 2, self.pred_len * self.c_out)

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

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        time_cov = self._build_time_covariates(x_mark_enc, x_enc)
        x = x_enc + self.time_feature_proj(time_cov)

        h_0 = x.new_zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        _, hn = self.gru(x, h_0)

        hn = hn.view(self.num_layers, 2, x.size(0), self.hidden_size)
        last_layer_hn = hn[-1]
        last_hidden = last_layer_hn.permute(1, 0, 2).reshape(x.size(0), -1)

        out = self.projection(last_hidden)
        out = out.reshape(out.shape[0], self.pred_len, self.c_out)
        return out
