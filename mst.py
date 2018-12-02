import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


def MST(content_relu, style_relu):
    '''
    relu1_2:[batch*64*height1*wideth1] (heigh1=wideth1=224)
    relu2_2:[batch*128*height2*wideth2] (heigh2=wideth2=122)
    relu3_3:[batch*256*height3*wideth3] (heigh3=wideth3=56)
    assume heigh1=height2*2=height3*4;       
    content_relu:[content_relu1_2, content_relu2_2, content_relu3_3]
    style_relu:[style_relu1_2, style_relu2_2, style_relu3_3]     
    '''
    factor2to1 = content_relu[0].size(2)//content_relu[1].size(2)
    factor3to1 = content_relu[0].size(2)//content_relu[2].size(2)

    upsample2to1 = nn.UpsamplingBilinear2d(factor2to1)
    upsample3to1 = nn.UpsamplingBilinear2d(factor3to1)

    content_relu[1] = upsample2to1(content_relu[1])
    style_relu[1] = upsample2to1(style_relu[1])

    content_relu[2] = upsample3to1(content_relu[2])
    style_relu[2] = upsample3to1(style_relu[2])

    content_relu = torch.stack(content_relu)
    style_relu = torch.stack(style_relu)

    content_relu = content_relu.view(content_relu.size(0), -1)
    style_relu = style_relu.view(style_relu.size(0), -1)

    content_relu_mean = torch.mean(content_relu, 1)
    content_relu_mean = content_relu_mean.unsqueeze(1).expand_as(content_relu)
    content_relu_zero_center = content_relu - content_relu_mean

    style_relu_mean = torch.mean(style_relu, 1)
    style_space_mean = torch.mean(style_relu, 0)
    style_relu_mean = style_relu_mean.unsqueeze(1).expand_as(style_relu)
    style_relu_zero_center = style_relu - style_relu_mean

    content_covariance, _, _ = torch.svd(content_relu_zero_center)
    feature_content = torch.rsqrt(content_covariance)*content_relu_zero_center

    style_covariance, _, _ = torch.svd(style_relu_zero_center)

    style_space_mean = style_space_mean.unsqueeze(1)
    transformed = torch.mm(torch.sqrt(style_covariance), feature_content) + torch.mm(style_space_mean,
                                                                                     torch.ones(1, style_space_mean.size(1)))
    return transformed


content_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
                                                        128, 112, 112), torch.rand(1, 256, 56, 56)]
style_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
                                                      128, 112, 112), torch.rand(1, 256, 56, 56)]
print(MST(content_relu, style_relu))
