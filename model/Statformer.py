import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class StatisticPositionMultiScaleAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_scales=3, dropout=0.1,
                 max_vars=64, use_vsr=True, use_pmsk=True,
                 learn_sigma=True, use_pos_emb=True, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_scales >= 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_scales = n_scales
        self.use_vsr = use_vsr
        self.use_pmsk = use_pmsk
        self.learn_sigma = learn_sigma
        self.use_pos_emb = use_pos_emb

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_pmsk:
            if use_pos_emb:
                pos_hidden = max(d_model // 4, 8)
                self.var_pos_mlp = nn.Sequential(
                    nn.Linear(1, pos_hidden),
                    nn.GELU(),
                    nn.Linear(pos_hidden, d_model),
                )
            if learn_sigma:
                self.log_sigma = nn.Parameter(torch.linspace(0.0, 3.0, n_scales))
            else:
                self.register_buffer('log_sigma', torch.linspace(0.0, 3.0, n_scales))
        if use_vsr and use_pmsk:
            route_hidden = max(d_model // 4, 8)
            self.router = nn.Sequential(
                nn.Linear(3, route_hidden),
                nn.GELU(),
                nn.Linear(route_hidden, n_scales),
            )

    def forward(self, queries, keys=None, values=None, attn_mask=None, tau=None, delta=None):
        x = queries
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head
        S = self.n_scales

        if not self.use_pmsk:
            qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
            out = self.proj(out)
            aux = {'route_weights': None, 'sigma': None}
            return out, aux
        if self.use_vsr:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x_diff = x.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
            stats = torch.cat([x_mean, x_std, x_diff], dim=-1)
            route_logits = self.router(stats)
            route_weights = F.softmax(route_logits, dim=-1)
        else:
            route_weights = x.new_full((B, N, S), 1.0 / S)

        if self.use_pos_emb:
            pos_idx = torch.arange(N, device=x.device).float().unsqueeze(-1)
            pos_emb = self.var_pos_mlp(pos_idx)
            x_pos = x + pos_emb.unsqueeze(0)
        else:
            x_pos = x

        qkv = self.qkv(x_pos).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        base_scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)

        pos_grid = torch.arange(N, device=x.device).float()
        dist = (pos_grid.unsqueeze(0) - pos_grid.unsqueeze(1)).abs()

        sigma = F.softplus(self.log_sigma)  # [K]
        kernel = torch.exp(-(dist ** 2).unsqueeze(0)
                           / (2 * sigma.view(-1, 1, 1) ** 2))  # [K, N, N]
        log_kernel = torch.log(kernel + 1e-8)  # [K, N, N]
        scores = base_scores.unsqueeze(-3) + log_kernel.unsqueeze(0).unsqueeze(0)
        # [B, H, K, N, N]
        attn = self.dropout(F.softmax(scores, dim=-1))
        outs = attn @ v.unsqueeze(-3)  # [B, H, K, N, Dh]

        route = route_weights.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)
        out = (outs * route).sum(dim=-3).permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.proj(out)

        aux = {
            'route_weights': route_weights.detach(),
            'sigma': F.softplus(self.log_sigma).detach(),
        }
        return out, aux

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy


        n_scales = int(getattr(configs, 'n_scales', 3))
        max_vars = int(getattr(configs, 'max_vars', 64))

        use_vsr = bool(int(getattr(configs, 'use_vsr', 1)))
        use_pmsk = bool(int(getattr(configs, 'use_pmsk', 1)))
        learn_sigma = bool(int(getattr(configs, 'learn_sigma', 1)))
        use_pos_emb = bool(int(getattr(configs, 'use_pos_emb', 1)))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    StatisticPositionMultiScaleAttention(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        n_scales=n_scales,
                        dropout=configs.dropout,
                        max_vars=max_vars,
                        use_vsr=use_vsr,
                        use_pmsk=use_pmsk,
                        learn_sigma=learn_sigma,
                        use_pos_emb=use_pos_emb,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]