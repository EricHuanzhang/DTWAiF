
from torch import nn, einsum
from einops import rearrange
import math, torch
from ..utils.ch_discover_loss import DynamicalContrastiveLoss


class BandwiseLayerNorm(nn.Module):
    """
    LayerNorm applied independently per wavelet band, so that a shared mean and
    variance cannot carry the scale of one band into another.
    """
    def __init__(self, sub_dims):
        super().__init__()
        self.sub_dims = sub_dims
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in sub_dims])

    def forward(self, x):
        chunks = torch.split(x, self.sub_dims, dim=-1)
        out_chunks = [norm(c) for norm, c in zip(self.norms, chunks)]
        return torch.cat(out_chunks, dim=-1)

class BandwiseLinear(nn.Module):
    """Block-diagonal linear map: one projection per band, no cross-band mixing."""
    def __init__(self, sub_dims_in, sub_dims_out, bias=True):
        super().__init__()
        self.sub_dims_in = sub_dims_in
        self.sub_dims_out = sub_dims_out
        self.linears = nn.ModuleList([
            nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(sub_dims_in, sub_dims_out)
        ])

    def forward(self, x):
        chunks = torch.split(x, self.sub_dims_in, dim=-1)
        out_chunks = [linear(c) for linear, c in zip(self.linears, chunks)]
        return torch.cat(out_chunks, dim=-1)

class BandwiseFeedForward(nn.Module):
    """Feed-forward block built from band-wise linear maps."""
    def __init__(self, sub_dims, mlp_dim, dropout=0.5):
        super().__init__()
        self.sub_dims = sub_dims
        total_dim = sum(sub_dims)
        self.nets = nn.ModuleList()
        for d in sub_dims:
            d_hidden = max(1, int(mlp_dim * (d / total_dim)))
            self.nets.append(nn.Sequential(
                nn.Linear(d, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d),
                nn.Dropout(dropout)
            ))

    def forward(self, x):
        chunks = torch.split(x, self.sub_dims, dim=-1)
        out_chunks = [net(c) for net, c in zip(self.nets, chunks)]
        return torch.cat(out_chunks, dim=-1)


class DualPreNorm(nn.Module):
    """Pre-normalisation wrapper for the two-track attention layer."""
    def __init__(self, dim, sub_dims, fn):
        super().__init__()
        self.norm_phys = BandwiseLayerNorm(sub_dims)
        self.norm_topo = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x_phys, x_topo, **kwargs):
        return self.fn(self.norm_phys(x_phys), self.norm_topo(x_topo), **kwargs)

class BandwisePreNorm(nn.Module):
    """Pre-normalisation wrapper using band-wise LayerNorm."""
    def __init__(self, sub_dims, fn):
        super().__init__()
        self.norm = BandwiseLayerNorm(sub_dims)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class c_Attention(nn.Module):
    """
    Cross-channel attention with asymmetric sources: queries and keys come from
    the context track, values from the content track.
    """
    def __init__(self, dim, sub_dims, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)

        total_dim = sum(sub_dims)
        self.inner_sub_dims = [max(1, int(inner_dim * (d / total_dim))) for d in sub_dims]
        self.inner_sub_dims[0] += inner_dim - sum(self.inner_sub_dims)

        self.to_v = BandwiseLinear(sub_dims, self.inner_sub_dims)
        self.to_out = nn.Sequential(
            BandwiseLinear(self.inner_sub_dims, sub_dims),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x_phys, x_topo, attn_mask=None):
        h = self.heads
        q = self.to_q(x_topo)
        k = self.to_k(x_topo)

        v = self.to_v(x_phys)

        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        if attn_mask is not None:
            q_norm = torch.norm(q, dim=-1, keepdim=True)
            k_norm = torch.norm(k, dim=-1, keepdim=True)
            norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
            def _mask(scores, attn_mask):
                large_negative = -1e9

                attention_mask = torch.where(attn_mask == 0, large_negative, 0.0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)

                return scores

            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn, dynamical_contrastive_loss

class c_Transformer(nn.Module):
    """
    Stack of cross-channel attention blocks. The context track is passed through
    each layer unchanged, so the routing prior cannot drift with depth.
    """
    def __init__(self, dim, sub_dims, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3,
                 temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DualPreNorm(dim, sub_dims,
                            c_Attention(dim, sub_dims, heads=heads, dim_head=dim_head, dropout=dropout,
                                        regular_lambda=regular_lambda, temperature=temperature)),
                BandwisePreNorm(sub_dims, BandwiseFeedForward(sub_dims, mlp_dim=mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_phys, x_topo, attn_mask=None):
        total_loss = 0
        all_attentions = []

        for attn_layer, ff in self.layers:
            x_n, current_attn, dcloss = attn_layer(x_phys, x_topo, attn_mask=attn_mask)

            all_attentions.append(current_attn)

            if dcloss is not None:
                total_loss += dcloss

            x_phys = x_n + x_phys
            x_phys = ff(x_phys) + x_phys

        if isinstance(total_loss, torch.Tensor):
            dcloss = total_loss / len(self.layers)
        else:
            dcloss = torch.tensor(0.0, device=x_phys.device)

        stacked_attn = torch.stack(all_attentions, dim=1)

        return x_phys, stacked_attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss
