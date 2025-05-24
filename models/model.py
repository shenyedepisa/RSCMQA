import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel
from models.imageModels import UNet
from models.textModels import seq2vec
from torchvision.models import resnet152, resnet101, vgg16, resnet18


class CDModel(nn.Module):
    def __init__(
            self,
            config,
            question_vocab,
            input_size,
            textHead,
            imageHead,
            trainText,
            trainImg,
    ):
        super(CDModel, self).__init__()
        self.config = config
        self.num_epochs = config["num_epochs"]
        self.question_vocab = question_vocab
        self.maskHead = config["maskHead"]
        self.textHead = textHead
        self.imageHead = imageHead

        self.fusion_in = config["FUSION_IN"]
        self.fusion_hidden = config["FUSION_HIDDEN"]
        self.num_classes = config["answer_number"]
        self.clipList = config["clipList"]
        self.vitList = config["vitList"]
        self.oneStep = config["one_step"]
        self.layer_outputs = {}
        self.attConfig = self.config["attnConfig"]
        self.imageModelPath = config["imageModelPath"]
        self.textModelPath = config["textModelPath"]

        from models import csmaBlock
        self.textEnhance = csmaBlock(config)
        # self.visualEnhance = csmaBlock(config)
        saveDir = config["saveDir"]

        if self.maskHead:
            if not self.oneStep:
                self.maskNet = torch.load(f"{saveDir}maskModel.pth")
                for param in self.maskNet.parameters():
                    param.requires_grad = False
            else:
                self.maskNet = UNet(n_channels=3, n_classes=3, bilinear=False)
                state_dict = torch.load(config["maskModelPath"])
                del state_dict["outc.conv.weight"]
                del state_dict["outc.conv.bias"]
                self.maskNet.load_state_dict(state_dict, strict=False)
        if self.imageHead == "siglip_512":
            siglip_model = AutoModel.from_pretrained(self.imageModelPath)
            self.imgModel = siglip_model.vision_model
            self.lineV = nn.Linear(768, 768)
        elif self.imageHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.imageModelPath)
            self.imgModel = clip.vision_model
            self.lineV = nn.Linear(768, 768)
        elif self.imageHead == "resnet_152":
            self.imgModel = resnet101(pretrained=True)
            extracted_layers = list(self.imgModel.children())
            extracted_layers = extracted_layers[0:8]  # Remove the last fc and avg pool
            self.imgModel = torch.nn.Sequential(*(list(extracted_layers)))
            output_size = (256 / 32) ** 2
            self.imgModel = torch.nn.Sequential(self.imgModel, torch.nn.Conv2d(2048, int(2048 / output_size), 1))
            self.lineV = nn.Linear(2048, 768)
        if self.textHead == "siglip_512":
            siglip_model = AutoModel.from_pretrained(self.textModelPath)
            self.textModel = siglip_model.text_model
        elif self.textHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.textModelPath)
            self.textModel = clip.text_model
            self.lineQ = nn.Linear(512, 768)
        elif self.textHead == 'skipthoughts':
            self.textModel = seq2vec.factory(question_vocab,
                                             {'arch': 'skipthoughts', 'dir_st': 'models/textModels/skip-thoughts',
                                              'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
            self.lineQ = nn.Linear(2400, 768)

        self.linear_classify1 = nn.Linear(self.fusion_in, self.fusion_hidden)
        self.linear_classify2 = nn.Linear(self.fusion_hidden, self.num_classes)
        self.dropout = torch.nn.Dropout(config["DROPOUT"])
        if not trainText:
            for param in self.textModel.parameters():
                param.requires_grad = False
        if not trainImg:
            for param in self.imglModel.parameters():
                param.requires_grad = False

    def forward(self, input_v, input_q):
        predict_mask = self.maskNet(input_v)
        m0 = predict_mask[:, 0, :, :].unsqueeze(1) / 255
        m1 = predict_mask[:, 1, :, :].unsqueeze(1) / 255
        m2 = predict_mask[:, 2, :, :].unsqueeze(1) / 255

        v = input_v + m2 + m1 + m0
        if self.imageHead in self.clipList:
            v = self.imgModel(pixel_values=v)["pooler_output"]
        elif self.imageHead == "resnet_152":
            v = self.imgModel(v).view(-1, 2048)
        v = self.dropout(v)
        v = self.lineV(v)
        v = nn.Tanh()(v)
        if self.textHead == "siglip-512":
            q = self.textModel(input_ids=input_q["input_ids"])["pooler_output"]
        elif self.textHead in self.clipList:
            q = self.textModel(**input_q)["pooler_output"]
        elif self.textHead == "skipthoughts":
            q, cells = self.textModel(input_q)
        q = self.dropout(q)
        q = self.lineQ(q)
        q = nn.Tanh()(q)

        t, kl = self.textEnhance(q, m0, m1)
        q = q + t
        x = torch.mul(v, q)
        x = nn.Tanh()(x)
        x = self.dropout(x)
        x = self.linear_classify1(x)
        x = nn.Tanh()(x)
        x = self.dropout(x)
        x = self.linear_classify2(x)

        return x, predict_mask, kl
