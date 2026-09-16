# This file is reproduced verbatim from the official CATCH implementation
# distributed with the TFB benchmark, and is included here so that the DTWAiF
# package is self-contained. All rights to the reused code remain with its
# original authors; see NOTICE for details.

'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:29
*
* @description: the implementation of the dynamical contrastive loss
'''

import torch


class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k  # k 是稀疏惩罚项的权重

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        # 增加 1e-8 防止除零导致 NaN (原版代码的隐患)
        cosine = (scores / (norm_matrix + 1e-8)).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask

        all_scores = torch.exp(cosine / self.temperature)

        # clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))
        # 为对数运算添加极小值epsilon保证稳定性
        # 聚类损失：由于完成了 Q-V 解耦，此时的聚类只会拉近 cA 的距离，极其安全
        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / (all_scores.sum(dim=-1) + 1e-8) + 1e-8)

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss
