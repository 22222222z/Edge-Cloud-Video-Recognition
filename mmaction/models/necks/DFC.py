'''
    SVD based Dynamic Feature Compression
'''
import torch
import torch.nn as nn
from mmaction.registry import MODELS

@MODELS.register_module()
class DFC(nn.Module):
    def __init__(self, top_ratio=1.0):
        super().__init__()
        self.top_ratio = top_ratio

    def forward(self, x):
        
        NT, C, H, W = x.size()
        x = x.reshape(NT//8, 8, C, H, W)

        # add noise for stable training
        epsilon = 1e-6
        x = x + epsilon * torch.randn_like(x)
        # x = x + epsilon * torch.eye(x.size(0), device=x.device)

        for i in range(x.shape[0]):

            video_i = x[i].squeeze(0)
            video_i = video_i.reshape(8*C, H*W)
            U, S, Vh = torch.linalg.svd(video_i, full_matrices=False)
            # print(f'U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}')

            top_singular_value = S.shape[0]
            for j in range(S.shape[0]):
                if torch.sum(S[:j]) > torch.sum(S) * self.top_ratio:
                    top_singular_value = j
                    break
            # print(f'before compression: {S.shape[0]}, after compression: {top_singular_value}')
            
            recons = U[:, :top_singular_value] @ torch.diag(S[:top_singular_value]) @ Vh[:top_singular_value, :] # torch.mm(torch.mm(U, torch.diag(S)), Vh)
            x[i] = recons.reshape(8, C, H, W).unsqueeze(0)
        
        x = x.reshape(NT, C, H, W)

        return x