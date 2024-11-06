import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel
from models.imageModels import UNet


class CDModel(nn.Module):
    def __init__(
            self,
            config,
            vocab_questions,
            vocab_answers,
            input_size,
            textHead,
            imageHead,
            trainText,
            trainImg,
    ):
        super(CDModel, self).__init__()
        self.config = config
        self.maskHead = config["maskHead"]
        self.textHead = textHead
        self.imageHead = imageHead
        self.imageModelPath = config["imageModelPath"]
        self.textModelPath = config["textModelPath"]
        self.fusion_in = config["FUSION_IN"]
        self.fusion_hidden = config["FUSION_HIDDEN"]
        self.vocab_answers = vocab_answers
        self.num_classes = config['answer_number']
        self.clipList = config["clipList"]
        self.vitList = config["vitList"]
        self.learnable_mask = config["learnable_mask"]
        self.imgOnly = config["img_only"]
        self.maskOnly = config["mask_only"]
        self.addMask = config["add_mask"]
        self.oneStep = config["one_step"]
        saveDir = config["saveDir"]
        from models import csmaBlock
        self.textEnhance = csmaBlock(config)
        if self.learnable_mask:
            self.weights = nn.Parameter(torch.randn(3, input_size, input_size) + 1)
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
        if self.imageHead == "siglip-512":
            siglip_model = AutoModel.from_pretrained(self.imageModelPath)
            self.imgModel = siglip_model.vision_model
        elif self.imageHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.imageModelPath)
            self.imgModel = clip.vision_model
            self.lineV = nn.Linear(768, 768)
            if self.addMask:
                clip1 = CLIPModel.from_pretrained(self.imageModelPath)
                self.maskModel = clip1.vision_model
                self.lineM = nn.Linear(768, 768)

        if self.textHead == "siglip-512":
            siglip_model = AutoModel.from_pretrained(self.textModelPath)
            self.textModel = siglip_model.text_model
        elif self.textHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.textModelPath)
            self.textModel = clip.text_model
            self.lineQ = nn.Linear(512, 768)

        self.attConfig = self.config["attnConfig"]
        self.linear_classify1 = nn.Linear(self.fusion_in, self.fusion_hidden)
        self.linear_classify2 = nn.Linear(self.fusion_hidden, self.num_classes)
        self.dropout = torch.nn.Dropout(config["DROPOUT"])
        if not trainText:
            for param in self.textModel.parameters():
                param.requires_grad = False

        if not trainImg:
            for param in self.imglModel.parameters():
                param.requires_grad = False

    def forward(self, input_v, input_q, mask=None):
        if self.imgOnly:
            v = self.imgModel(pixel_values=input_v)["pooler_output"]
            v = self.dropout(v)
            v = self.lineV(v)
            v = nn.Tanh()(v)
            m = self.maskModel(pixel_values=mask)["pooler_output"]
            m = self.dropout(m)
            m = self.lineM(m)
            m = nn.Tanh()(m)
            if self.textHead == "siglip-512":
                q = self.textModel(input_ids=input_q["input_ids"])["pooler_output"]
            elif self.textHead in self.clipList:
                q = self.textModel(**input_q)["pooler_output"]
                q = self.dropout(q)
                q = self.lineQ(q)
                q = nn.Tanh()(q)
            else:
                q = self.textModel(input_q)
            predict_mask = mask
            vm = torch.cat((v, m), dim=1)
            x = self.crossAtt1(vm.unsqueeze(1), q.unsqueeze(1))
            x = nn.Tanh()(x.squeeze(1))
            x = self.dropout(x)
            x = self.linear_classify1(x)
            x = nn.Tanh()(x)
            x = self.dropout(x)
            x = self.linear_classify2(x)
        elif self.addMask:
            predict_mask = self.maskNet(input_v)
            if self.learnable_mask:
                predict_mask = predict_mask * self.weights
            m0 = predict_mask[:, 0, :, :].unsqueeze(1)
            m1 = predict_mask[:, 1, :, :].unsqueeze(1)
            m2 = predict_mask[:, 2, :, :].unsqueeze(1)
            v = input_v + m2
            t = self.textEnhance(m0, m1)
            v = self.imgModel(pixel_values=v)["pooler_output"]
            v = self.dropout(v)
            v = self.lineV(v)
            v = nn.Tanh()(v)
            if self.textHead == "siglip-512":
                q = self.textModel(input_ids=input_q["input_ids"])["pooler_output"]
            elif self.textHead in self.clipList:
                q = self.textModel(**input_q)["pooler_output"]
                q = self.dropout(q)
                q = self.lineQ(q)
                q = nn.Tanh()(q)
            else:
                q = self.textModel(input_q)
            q = q + t
            x = torch.mul(v, q)
            x = nn.Tanh()(x)
            x = self.dropout(x)
            x = self.linear_classify1(x)
            x = nn.Tanh()(x)
            x = self.dropout(x)
            x = self.linear_classify2(x)
        else:
            x = 'undefined'
            predict_mask = 'undefined'
        return x, predict_mask
