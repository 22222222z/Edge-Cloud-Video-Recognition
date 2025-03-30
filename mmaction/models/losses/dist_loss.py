'''
    Huang et al. Knowledge Distillation from A Stronger Teacher. NeurIPS 2022
    paper: https://arxiv.org/pdf/2205.10536
    code: https://github.com/hunto/DIST_KD
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()

def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@MODELS.register_module()
class RelationDistLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0, reduction='mean'):
        super(RelationDistLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        # print(z_s.shape, z_t.shape)
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)

        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)

        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        # print("RelationDistLoss", inter_loss, intra_loss)

        return kd_loss