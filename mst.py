import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

def wst(content_relu, style_relu):
    '''
    relu1_2:[64*height1*wideth1]
    relu2_2:[128*height2*wideth2]
    relu3_3:[256*height3*wideth3]
    assume heigh1=height2*2=height3          
    content_relu:[content_relu1_2, content_relu2_2, content_relu3_3]
    style_relu:[style_relu1_2, style_relu2_2, style_relu3_3]     
    '''
    factor2to1 = content_relu[0].size(2)//content_relu[1].size(2)
    factor3to1 = content_relu[0].size(2)//content_relu[3].size(2)

    upsample2to1 = nn.UpsamplingBilinear2d(factor2to1)
    upsample3to1 = nn.UpsamplingBilinear2d(factor3to1)

    content_relu[1] = upsample2to1(content_relu[1])
    style_relu[1] = upsample2to1(style_relu[1])

    content_relu[2] = upsample3to1(content_relu[2])
    style_relu[2] = upsample3to1(style_relu[2])

    content_relu = torch.stack(content_relu)
    style_relu = torch.stack(style_relu)
    
    content_relu_mean = torch.mean(content_relu, 1)
    content_relu_mean = content_relu_mean.unsqueeze(1).expand_as(content_relu)
    content_relu = content_relu - content_relu_mean

    #   torch.svd(content_relu, )
