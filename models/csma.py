import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from .modules import CrossAttention, MLP
import torch.nn.functional as F


def KL(A, B):
    prob_A = torch.clamp(F.softmax(A, dim=-1), min=1e-9, max=1.0)
    prob_B = torch.clamp(F.softmax(B, dim=-1), min=1e-9, max=1.0)
    kl_div = F.kl_div(prob_A.log(), prob_B, reduction='batchmean')
    return kl_div


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

        self.cnnEncoder1 = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder1.fc.in_features
        self.cnnEncoder1.fc = torch.nn.Linear(num_ftrs, self.embed_size)
        self.cnnEncoder1.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.cnnEncoder2 = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder2.fc.in_features
        self.cnnEncoder2.fc = torch.nn.Linear(num_ftrs, self.embed_size)
        self.cnnEncoder2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        self.crossAtt1 = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.crossAtt2 = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.crossAtt3 = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.out = nn.Linear(int(self.embed_size * 2), self.embed_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=768)

    def forward(self, text, source, target):
        s1 = self.cnnEncoder(source)
        t1 = self.cnnEncoder(target)
        sim = KL(s1, t1)
        s2 = self.cnnEncoder1(source)
        t2 = self.cnnEncoder2(target)
        dis_sim = 1/(KL(s2, t2) + 1e-9)
        # s = source
        # t = target
        att1 = self.crossAtt(text.unsqueeze(1), s1.unsqueeze(1)).squeeze(1)
        att2 = self.crossAtt1(text.unsqueeze(1), t1.unsqueeze(1)).squeeze(1)
        att3 = self.crossAtt2(text.unsqueeze(1), s2.unsqueeze(1)).squeeze(1)
        att4 = self.crossAtt3(text.unsqueeze(1), t2.unsqueeze(1)).squeeze(1)
        output = att1+att2 + self.mlpT(att1+att2)
        output1 = att3+att4 + self.mlpT(att3+att4)

        return output+output1, sim + dis_sim
