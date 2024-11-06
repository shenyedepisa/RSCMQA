import torch
import torch.nn as nn
from models.imageModels.milesial_UNet import UNet


class maskModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(maskModel, self).__init__()
        self.config = config
        self.maskNet = UNet(n_channels=3, n_classes=3, bilinear=False)
        state_dict = torch.load(config["maskModelPath"])
        del state_dict["outc.conv.weight"]
        del state_dict["outc.conv.bias"]
        self.maskNet.load_state_dict(state_dict, strict=False)

    def forward(self, input_v):
        predict_mask = self.maskNet(input_v)
        return predict_mask
