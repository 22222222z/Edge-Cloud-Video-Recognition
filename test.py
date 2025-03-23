import torch
import torch.nn.functional as F

a = torch.randn(3, 4)
normalized_a = F.normalize(a, p=2, dim=1)
print(normalized_a)