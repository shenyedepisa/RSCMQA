import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from .modules import CrossAttention, MLP

class csmaBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(csmaBlock, self).__init__()
        self.config = config
        self.embed_size = self.config["FUSION_IN"]
        self.cnnEncoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder.fc.in_features
        self.cnnEncoder.fc = torch.nn.Linear(num_ftrs, self.embed_size)
        self.cnnEncoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.attConfig = self.config["attnConfig"]
        self.mlpS = MLP(
            self.attConfig["embed_size"],
            int(self.attConfig["embed_size"] * self.attConfig["mlp_ratio"]),
            self.attConfig["embed_size"],
            self.attConfig["attn_dropout"],
        )
        self.mlpT = MLP(
            self.attConfig["embed_size"],
            int(self.attConfig["embed_size"] * self.attConfig["mlp_ratio"]),
            self.attConfig["embed_size"],
            self.attConfig["attn_dropout"],
        )
        self.crossAtt = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.out = nn.Linear(int(self.embed_size * 2), self.embed_size)

    def forward(self, source, target):
        s = self.cnnEncoder(source)
        t = self.cnnEncoder(target)
        att = self.crossAtt(s.unsqueeze(1), t.unsqueeze(1)).squeeze(1)
        s = s + self.mlpS(att)
        t = s + self.mlpT(att)
        output = torch.cat((s, t), dim=1)
        output = self.out(output)
        return output
