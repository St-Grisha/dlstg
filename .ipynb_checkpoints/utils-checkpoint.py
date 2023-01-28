import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).clone().view(-1, 1, 1)
        self.std = torch.tensor(std).clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
def get_input_optimizer(input_img: torch.Tensor) -> torch.optim.Optimizer:
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def image_loader(image_name):
    loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)), 
        transforms.ToTensor(),
    ]
    )
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

