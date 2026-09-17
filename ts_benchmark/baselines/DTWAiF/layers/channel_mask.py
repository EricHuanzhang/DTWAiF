
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax


class channel_mask_generator(torch.nn.Module):
    """
    Generate a sparse cross-channel attention mask.

    A Gumbel-softmax relaxation makes the discrete keep/drop decision
    differentiable, so the mask is learned jointly with the reconstruction.
    """
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(torch.nn.Linear(input_size, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        distribution_matrix = self.generator(x)

        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag

        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        flatten_matrix = torch.clamp(flatten_matrix, min=1e-6, max=1.0 - 1e-6)

        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        if self.training:
            resample_matrix = gumbel_softmax(new_matrix, hard=True)
        else:
            hard_mask = (new_matrix[..., 0] > new_matrix[..., 1]).float()
            resample_matrix = torch.stack([hard_mask, 1.0 - hard_mask], dim=-1)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)

        return resample_matrix
