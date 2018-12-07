import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert(len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)
                       )/content_std.expand(size)
    return normalized_feat*style_std.expand(size)+style_mean.expand(size)


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

    transformed = adaptive_instance_normalization(content_relu, style_relu)

    result_relu1_2 = transformed[:, :64, :, :]
    result_relu2_2 = transformed[:, 64:192, :, :]
    result_relu3_3 = transformed[:, 192:, :, :]
    return result_relu1_2, result_relu2_2, result_relu3_3
