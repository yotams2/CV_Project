"""Define your architecture here."""
import torch
from models import SimpleNet
import torchvision


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = torchvision.models.shufflenet_v2_x0_5(weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Sequential(torch.nn.Linear(1024, 576),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(576, 256),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(256, 64),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(64, 2))
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
