import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.c_out = configs.c_out
        self.time_feat_dim = 4
        self.time_feature_proj = nn.Linear(self.time_feat_dim, self.input_size)

        rnn_dropout = configs.dropout if self.num_layers > 1 else 0
        self.encoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.decoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_size, self.c_out),
        )

    def _build_time_covariates(self, x_mark, reference):
        b, l, _ = reference.shape
        if x_mark is None:
            return reference.new_zeros(b, l, self.time_feat_dim)

        x_mark = x_mark[..., :self.time_feat_dim]
        if x_mark.size(1) > l:
            x_mark = x_mark[:, :l, :]
        elif x_mark.size(1) < l:
            pad = x_mark.new_zeros(b, l - x_mark.size(1), x_mark.size(-1))
            x_mark = torch.cat([x_mark, pad], dim=1)

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
        enc_time_cov = self._build_time_covariates(x_mark_enc, x_enc)
        enc_input = x_enc + self.time_feature_proj(enc_time_cov)

        h_0 = enc_input.new_zeros(self.num_layers, enc_input.size(0), self.hidden_size)
        _, hidden = self.encoder(enc_input, h_0)

        if x_dec is None:
            x_dec = x_enc[:, -1:, :].repeat(1, self.label_len + self.pred_len, 1)
        dec_time_cov = self._build_time_covariates(x_mark_dec, x_dec)
        dec_input = x_dec + self.time_feature_proj(dec_time_cov)

        dec_out, _ = self.decoder(dec_input, hidden)
        output = self.output_proj(dec_out)
        return output
