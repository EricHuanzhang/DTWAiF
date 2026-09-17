import torch
import torch.nn as nn
import ptwt
import pywt
import torch.nn.functional as F
from einops import rearrange
from typing import List

from ts_benchmark.baselines.DTWAiF.layers.cross_channel_bandwise_Transformer import BandwiseLayerNorm, c_Transformer
from ts_benchmark.baselines.DTWAiF.layers.channel_mask import channel_mask_generator


class ScaleAwareWaveletEncoder(nn.Module):
    """

    Per-band encoder producing the two tracks.

    Each wavelet band gets its own projection, so bands are never mixed by a shared
    linear layer. The encoder returns (content, context): the content embedding
    spans all bands, the context embedding carries the approximation band cA alone.

    """
    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(ScaleAwareWaveletEncoder, self).__init__()
        num_bands = len(coeff_shapes)

        self.topo_encoder = nn.Sequential(
            nn.Linear(coeff_shapes[0], d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.topo_norm = nn.LayerNorm(d_model)

        self.sub_dims = [d_model // num_bands] * num_bands
        self.sub_dims[0] += d_model % num_bands

        self.phys_branches = nn.ModuleList()
        for length, sub_dim in zip(coeff_shapes, self.sub_dims):
            branch = nn.Sequential(
                nn.Linear(length, sub_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.phys_branches.append(branch)

        self.fusion_norm = BandwiseLayerNorm(self.sub_dims)

    def forward(self, coeffs_list):
        x_topo = self.topo_norm(self.topo_encoder(coeffs_list[0]))

        phys_subs = []
        for coeff, branch in zip(coeffs_list, self.phys_branches):
            phys_subs.append(branch(coeff))

        x_phys_concat = torch.cat(phys_subs, dim=-1)
        x_phys = self.fusion_norm(x_phys_concat)

        return x_phys, x_topo


class ScaleAwareWaveletDecoder(nn.Module):
    """
    Project the latent representation back to a list of wavelet coefficients,
    one head per band, mirroring the encoder.
    """
    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(ScaleAwareWaveletDecoder, self).__init__()
        num_bands = len(coeff_shapes)

        self.sub_dims = [d_model // num_bands] * num_bands
        self.sub_dims[0] += d_model % num_bands

        self.branch_decoders = nn.ModuleList()
        for length, sub_dim in zip(coeff_shapes, self.sub_dims):
            branch = nn.Sequential(
                nn.Linear(sub_dim, sub_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(sub_dim, length)
            )
            self.branch_decoders.append(branch)

    def forward(self, latent_repr):
        recon_coeffs_list = []
        cursor = 0

        for sub_dim, decoder in zip(self.sub_dims, self.branch_decoders):
            band_latent = latent_repr[..., cursor: cursor + sub_dim]

            recon_coeffs_list.append(decoder(band_latent))
            cursor += sub_dim

        return recon_coeffs_list

class DTWAiFModel(nn.Module):
    """

    Dual-track wavelet Transformer reconstructor.

        DWT -> per-band encoding -> masked cross-channel attention -> per-band
        decoding -> inverse DWT

    Attention takes queries and keys from the context track and values from the
    content track, and the channel mask is generated from the context track too, so
    channel routing follows slow structure rather than high-frequency noise.

    """
    def __init__(self, configs, discrete_mask=None):
        super(DTWAiFModel, self).__init__()

        self.seq_len = configs.seq_len
        self.c_in = configs.c_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.wavelet = pywt.Wavelet('db4')
        self.level = pywt.dwt_max_level(self.seq_len, self.wavelet.dec_len)

        self.mask_generator = channel_mask_generator(input_size=self.d_model, n_vars=self.c_in)

        dummy_input = torch.zeros(1, 1, self.seq_len)
        dummy_coeffs = ptwt.wavedec(dummy_input, self.wavelet, level=self.level, mode='symmetric')
        self.coeff_shapes = [coeff.shape[-1] for coeff in dummy_coeffs]

        self.wave_encoder = ScaleAwareWaveletEncoder(self.d_model, self.coeff_shapes, configs.dropout)
        self.encoder = c_Transformer(
            dim=configs.d_model,
            sub_dims=self.wave_encoder.sub_dims,
            depth=configs.e_layers,
            heads=configs.n_heads,
            dim_head=configs.head_dim,
            mlp_dim=configs.d_ff,
            dropout=configs.dropout,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature
        )

        self.reconstruction_projector = ScaleAwareWaveletDecoder(
            d_model=self.d_model,
            coeff_shapes=self.coeff_shapes,
            dropout_rate=configs.dropout
        )

    def forward(self, x):
        """
        :param x: [batch, seq_len, channels]
        :return: (reconstruction, attention weights, contrastive loss,
            channel mask, wavelet coefficients)
        """
        x_permuted = x.permute(0, 2, 1)

        wavelet_coeffs_list = ptwt.wavedec(x_permuted, self.wavelet, level=self.level, mode='symmetric')

        fused_embedding, cA_topology_embedding = self.wave_encoder(wavelet_coeffs_list)

        if self.c_in <= 1:
            channel_mask = None
        else:
            channel_mask = self.mask_generator(cA_topology_embedding)

        latent_repr, attn_weights, dcloss = self.encoder(
            x_phys=fused_embedding,
            x_topo=cA_topology_embedding,
            attn_mask=channel_mask
        )

        recon_coeffs_list = self.reconstruction_projector(latent_repr)

        out_reconstructed = ptwt.waverec(recon_coeffs_list, self.wavelet)

        if out_reconstructed.shape[-1] > self.seq_len:
            excess = out_reconstructed.shape[-1] - self.seq_len
            start = excess // 2
            end = start + self.seq_len
            out_reconstructed = out_reconstructed[..., start:end]

        out_reconstructed = out_reconstructed.permute(0, 2, 1)
        out_final = out_reconstructed

        return out_final, recon_coeffs_list, dcloss, attn_weights, channel_mask

    def reconstruct_from_scaled(self, x_scaled, channel_drop_mask=None, fill_values=None):
        """Reconstruct with a subset of channels masked out, for counterfactual analysis."""
        if channel_drop_mask is not None:
            if channel_drop_mask.dtype != torch.bool:
                channel_drop_mask = channel_drop_mask.bool()
            if channel_drop_mask.dim() == 1:
                channel_drop_mask = channel_drop_mask.unsqueeze(0).expand(x_scaled.size(0), -1)

            if fill_values is None:
                raise ValueError("counterfactual masking requires fill_values "
                                 "(typically the per-channel median of the scaled training set)")

            fv = fill_values.view(1, 1, -1).expand(x_scaled.size(0), x_scaled.size(1), -1)
            x_scaled = torch.where(channel_drop_mask.unsqueeze(1), fv, x_scaled)

        x_permuted = x_scaled.permute(0, 2, 1)
        wavelet_coeffs_list = ptwt.wavedec(x_permuted, self.wavelet, level=self.level, mode='symmetric')

        fused_embedding, cA_topology_embedding = self.wave_encoder(wavelet_coeffs_list)
        channel_mask = self.mask_generator(cA_topology_embedding)

        latent_repr, attn_weights, dcloss = self.encoder(
            x_phys=fused_embedding,
            x_topo=cA_topology_embedding,
            attn_mask=channel_mask
        )

        recon_coeffs_list = self.reconstruction_projector(latent_repr)

        out_reconstructed = ptwt.waverec(recon_coeffs_list, self.wavelet)
        if out_reconstructed.shape[-1] > self.seq_len:
            out_reconstructed = out_reconstructed[..., :self.seq_len]

        out_final = out_reconstructed.permute(0, 2, 1)
        return out_final, recon_coeffs_list, dcloss, attn_weights, channel_mask
