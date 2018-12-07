import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple
def whiten_and_color(cF,sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).float().cuda()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

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

    # content_relu[2] = upsample3to1(content_relu[2])
    # style_relu[2] = upsample3to1(style_relu[2])

    # content_relu = torch.cat(content_relu, dim=1)
    # style_relu = torch.cat(style_relu, dim=1)

    content_relu = content_relu[-1]
    style_relu = style_relu[-1]

    # because, torch.svd() can only apply on 2D-tensor, so, we apply following step on each sample
    # and flatten width,height 2D into width*height 1D
    result_relu1_2, result_relu2_2, result_relu3_3 = [], [], []
    for sample in range(content_relu.size(0)):
        # content = content_relu[sample]
        # content = content.view(content.size(0), -1)

        # style = style_relu[sample]
        # style = style.view(style.size(0), -1)

        # content_mean = torch.mean(content, 1, keepdim=True)
        # content_0_center = content-content_mean

        # style_mean = torch.mean(style, 1, keepdim=True)
        # style_0_center = style - style_mean

        # content_covariance = torch.mm(content_0_center, content_0_center.t()) + torch.eye(content.size(0)).float().cuda()
        # # print(content_covariance)
        # content_u, content_e, content_v = torch.svd(
        #     content_covariance, some=True)
        
        # style_covariance = torch.mm(style_0_center, style_0_center.t())+torch.eye(style.size(0)).float().cuda()
        # # print(style_covariance)
        # style_u, style_e, style_v = torch.svd(style_covariance, some=False)

        # step = torch.mm(content_v.t(), content_0_center)
        # step = torch.mm(torch.diag(torch.rsqrt(content_e)),
        #                 step)  # may go wrong

        # step = torch.mm(content_v, step)
        # step = torch.mm(style_v.t(), step)
        # step = torch.mm(torch.diag(torch.sqrt(style_e)), step)
        # step = torch.mm(style_v, step)

        # transformed = step + style_mean.expand_as(content)

        content = content_relu[sample]
        style = style_relu[sample]
        C = content.size(0)
        cFView = content.view(C,-1)
        sFView = style.view(C,-1)
        transformed = whiten_and_color(cFView, sFView)
        # print("t size: ", transformed.size())

        # result_relu1_2.append(transformed[:64].view(64, 224, 224).unsqueeze(0))
        # result_relu2_2.append(
        #     transformed[64:192].view(128, 224, 224).unsqueeze(0))
        result_relu3_3.append(
            transformed.view(256, 56, 56).unsqueeze(0))
    # result_relu1_2 = torch.cat(result_relu1_2)
    # result_relu2_2 = torch.cat(result_relu2_2)
    result_relu3_3 = torch.cat(result_relu3_3)
    return result_relu3_3, result_relu3_3, result_relu3_3


# content_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
#                                                         128, 112, 112), torch.rand(1, 256, 56, 56)]
# style_relu = [torch.rand(1, 64, 224, 224), torch.rand(1,
#                                                       128, 112, 112), torch.rand(1, 256, 56, 56)]
# print(MST(content_relu, style_relu))
