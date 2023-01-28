import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from utils import Normalization, get_input_optimizer, image_loader
from losses import ContentLoss, StyleLoss


device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def get_style_model_and_losses(
    cnn: nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    content_layers = content_layers_default,
    style_layers = style_layers_default,
):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[: (i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(
    cnn: nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    input_img: torch.Tensor,
    num_steps: int = 15,
    style_weight: float = 1000000.0,
    content_weight: float = 1.0,
) -> torch.Tensor:
    print("Building the style transfer model...")

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )
    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")

    style_score_hist = []
    content_score_hist = []
    for step in range(num_steps):

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = style_weight * sum([s.loss for s in style_losses])
            content_score = content_weight * sum([c.loss for c in content_losses])

            loss = style_score + content_score
            style_score_hist.append(style_score)
            content_score_hist.append(content_score)
            loss.backward()

            return loss

        # specifics of LBFGS optimizer
        optimizer.step(closure)

        if (step) % 5 == 0:
            print(f"Step number: {step + 1}")
            print(
                f"Style Loss : {style_score_hist[-1].item():.4f} Content Loss: {content_score_hist[-1].item():.4f}"
            )

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def gen_output(filename):
    style_img = image_loader(f"style_{filename}.jpg")
    content_img = image_loader(f"content_{filename}.jpg")
    input_img = content_img.clone()
    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        style_img,
        content_img,
        input_img,
        style_weight=500_000,
    )
    save_image(output, f'result_{filename}.png')
    return 1
    
