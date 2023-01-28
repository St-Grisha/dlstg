import torch.nn as nn
import torch.nn.functional as F
import torch

class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x



def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    (
        batch_size,
        num_of_fm,
        h,
        w,
    ) = x.size() 
    features = x.view(batch_size * num_of_fm, h * w) 
    G = torch.matmul(features, features.t())
    
    return G.div(batch_size * num_of_fm * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x