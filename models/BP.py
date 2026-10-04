import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Traditional BP network baseline (single hidden-layer MLP).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model

        self.time_feat_dim = 4
        self.time_feature_proj = nn.Linear(self.time_feat_dim, self.enc_in)

        self.model = nn.Sequential(
            nn.Linear(self.seq_len * self.enc_in, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.pred_len * self.c_out),
        )

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
        b = x_enc.shape[0]

        time_cov = self._build_time_covariates(x_mark_enc, x_enc)
        x_enc = x_enc + self.time_feature_proj(time_cov)

        x = x_enc.reshape(b, -1)
        output = self.model(x)
        output = output.reshape(b, self.pred_len, self.c_out)
        return output
