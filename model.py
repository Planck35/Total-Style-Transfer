import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        return vgg_outputs(*results)


class Decoder(nn.Module):
    def __init__(self,d):
        super(Decoder,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.conv11.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv11.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.conv12.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv12.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.conv13.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv13.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.conv14.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv14.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.conv15.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.conv16.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.conv17.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.conv18.weight = torch.nn.Parameter(d.get(25).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(25).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)
        self.conv19.weight = torch.nn.Parameter(d.get(28).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(28).bias.float())



    def forward(self,x, relu1_2, relu2_2, relu3_3):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out) + relu3_3 # 256
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out) + relu2_2 #128
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out) + relu1_2  #64
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out
