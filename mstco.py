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

    content = torch.cat(content_relu, dim=1)
    style = torch.cat(style_relu, dim=1)

    content = content.view(content.size(0), content.size(1), -1)
    content_mean = torch.mean(content, 2, keepdim=True)

    content_0_center = content - content_mean
    # print(content_0_center.size())
    content_covariance = torch.bmm(content_0_center, content_0_center.transpose(1,2)).div(content_0_center.size()[2]-1)
    print(content_covariance >= 0)

    style = style.view(style.size(0), style.size(1), -1)
    style_mean = torch.mean(style, 2, keepdim=True)

    style_0_center = style - style_mean
    style_covariance = torch.bmm(style_0_center, style_0_center.transpose(1,2)).div(style_0_center.size()[2]-1)

    step = torch.bmm(content_covariance.pow(-0.5), content_0_center)
    

    transformed = torch.bmm(style_covariance.pow(0.5),step)+style_mean
    # print(transformed)

    return transformed[:,:64,:].view(-1,64,224, 224), transformed[:,64:192,:].view(-1,128,224, 224), transformed[:,192:,:].view(-1,256,224, 224)


# content_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
#                                                         128, 112, 112), torch.rand(1, 256, 56, 56)]
# style_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
#                                                       128, 112, 112), torch.rand(1, 256, 56, 56)]
# print(MST(content_relu, style_relu))
