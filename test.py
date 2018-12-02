import torch
import torch.nn as nn
from collections import namedtuple
from torchvision.models import vgg16_bn


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        features = list(vgg16_bn(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15}:
                results.append(x)

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        return vgg_outputs(*results)


encode = Encoder()
randomInput = torch.rand(1, 3, 224, 224)
randomOutput = encode(randomInput)
print(vgg)
