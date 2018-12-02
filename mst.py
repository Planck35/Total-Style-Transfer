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

    upsample2to1 = nn.UpsamplingBilinear2d(scale_factor=factor2to1)
    upsample3to1 = nn.UpsamplingBilinear2d(scale_factor=factor3to1)

    content_relu[1] = upsample2to1(content_relu[1])
    style_relu[1] = upsample2to1(style_relu[1])

    content_relu[2] = upsample3to1(content_relu[2])
    style_relu[2] = upsample3to1(style_relu[2])

    content_relu = torch.cat(content_relu, dim=1)
    style_relu = torch.cat(style_relu, dim=1)

    # because, torch.svd() can only apply on 2D-tensor, so, we apply following step on each sample
    # and flatten width,height 2D into width*height 1D
    result_relu1_2, result_relu2_2, result_relu3_3 = [], [], []
    for sample in range(content_relu.size(0)):
        content = content_relu[sample]
        content = content.view(content.size(0), -1)
        # print(content.size())
        style = style_relu[sample]
        style = style.view(style.size(0), -1)
        # print(style.size())
        content_mean = torch.mean(content, 1, keepdim=True)
        content_zero_center = content-content_mean
        # print(content_mean.size())
        # print(content_zero_center.size())
        style_mean = torch.mean(style, 1, keepdim=True)
        style_zero_center = style - style_mean
        # print(style_mean.size())
        # print(style_zero_center.size())
        content_covariance, _, _ = torch.svd(content_zero_center)
        # print(content_covariance.size())
        feature_content = torch.mm(torch.rsqrt(
            content_covariance), content_zero_center)
        # print(feature_content.size())
        style_covariance, _, _ = torch.svd(style_zero_center)

        style_space_mean = torch.mean(style, 1, keepdim=True)
        # print(style_space_mean.size())
        transformed = torch.mm(torch.sqrt(style_covariance), feature_content) + torch.mm(style_space_mean,
                                                                                         torch.ones(1, content.size(1)))
        print(transformed)
        result_relu1_2.append(transformed[:64].view(64, 224, 224).unsqueeze(0))
        result_relu2_2.append(
            transformed[64:192].view(128, 224, 224).unsqueeze(0))
        result_relu3_3.append(
            transformed[192:].view(256, 224, 224).unsqueeze(0))
    result_relu1_2 = torch.cat(result_relu1_2)
    result_relu2_2 = torch.cat(result_relu2_2)
    result_relu3_3 = torch.cat(result_relu3_3)
    return result_relu1_2, result_relu2_2, result_relu3_3


content_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
                                                        128, 112, 112), torch.rand(1, 256, 56, 56)]
style_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
                                                      128, 112, 112), torch.rand(1, 256, 56, 56)]
print(MST(content_relu, style_relu))
