import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.misc
from torch.utils.serialization import load_lua
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import Encoder, Decoder
from img_loader import get_data_loader
from mst import MST
from torch import nn


MAX_EPOCH = 150
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
CONTENT_PATH = "./data/content_img/"
STYLE_PATH = "./data/style_img/"

def calc_mean_cov(matrix):
    '''[compute the covariance of image]
    
    Arguments:
        cF {[Tensor]} -- [size:batch*3*224*224]
    
    Returns:
        [type] -- [description]
    '''
    
    # b*c*(h*w)
    matrix = matrix.view(matrix.size(0), matrix.size(1),-1)
    matrix_mean = torch.mean(matrix, 2, keepdim=True)
    matrix_0_center = matrix - matrix_mean
    #b*[c*(h*w), (h*w)*c] = b*(c*c)
    matrix_covariance = torch.bmm(matrix_0_center, matrix_0_center.transpose(1,2))

    return matrix_mean, matrix_covariance

def calc_style_loss(cF, sF, loss_fn):
    m1, c1 = calc_mean_cov(cF)
    m2, c2 = calc_mean_cov(sF)
    return loss_fn(m1, m2) + loss_fn(c1, c2)

def upsample_and_cat(content_relu):
    factor2to1 = content_relu[0].size(2)//content_relu[1].size(2)
    factor3to1 = content_relu[0].size(2)//content_relu[2].size(2)
    upsample2to1 = nn.UpsamplingBilinear2d(scale_factor=factor2to1)
    upsample3to1 = nn.UpsamplingBilinear2d(scale_factor=factor3to1)
    content_relu[1] = upsample2to1(content_relu[1])
    content_relu[2] = upsample3to1(content_relu[2])
    return torch.cat(content_relu, dim=1)

class TotalSytle():
    def __init__(self):

        self.train_loader = get_data_loader(
            content_path = CONTENT_PATH,
            style_path = STYLE_PATH,
            batch_size = BATCH_SIZE,
            small_test = True
        )

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mse_loss = torch.nn.MSELoss()
        # self.style_loss = StyleLoss() ## TODO: Complete Styleloss
        # self.content_loss = ContentLoss() ## TODO: Complete ContentLoss
        print ("----------------------Model is loaded----------------------------")
        parameters = self.decoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

        self.use_gpu = True
        if self.use_gpu:
            print ("----------------------GPU is used to train---------------------------")
            self.encoder.cuda()
            self.decoder.cuda()
            self.mse_loss.cuda()
        else:
            print ("----------------------CPU is used to train----------------------------")

        
        self.alpha = 0.5 #the weight of content loss and style loss
        self.beta = 1 # the weight of inter_scale loss and inter_scale loss. 1: only use loss_intra_scale

    def train(self):
        
        for epoch in range(MAX_EPOCH):
            total_loss = 0
            for batch_id, (style_imgs, content_imgs) in enumerate(self.train_loader):
                style_imgs = style_imgs.cuda()
                content_imgs = content_imgs.cuda()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                #Parse the style_imgs and content_imgs into encoder
                encoded_style, output_style = self.encoder(style_imgs)
                encoded_content, output_content = self.encoder(content_imgs)
                e_loss = encoded_content[-1]
                encoded_style_save = encoded_style.copy()
                #Compute the MST transformed relu
                relu1_2, relu2_2, relu3_3 = MST(encoded_style, encoded_content)
                # print(relu1_2.size(), relu2_2.size(), relu3_3.size())
                #Skip connection with decoder
                stylized_img = self.decoder(relu1_2, relu2_2, relu3_3)
                # print(stylized_img.size())

                #Extract the features of generated stylized img
                encoded_stylized, _ = self.encoder(stylized_img)

                #compute the loss between stylized imgs and content imgs
                #use only relu3_3 to as the 'content' of an img
                loss_c = self.mse_loss(encoded_stylized[-1], e_loss)

                #compute the loss between stylized imgs and style imgs
                # intra scale loss
                loss_intra_scale = calc_style_loss(encoded_stylized[0], encoded_style_save[0], self.mse_loss)
                for i in range(1, 3):
                    loss_intra_scale += calc_style_loss(encoded_stylized[i], encoded_style_save[i], self.mse_loss)

                # inter scale loss
                encoded_stylized = upsample_and_cat(encoded_stylized)
                encoded_content = upsample_and_cat(encoded_content)
                loss_inter_sacle = calc_style_loss(encoded_stylized, encoded_content, self.mse_loss)

                #weighted sum of inter_scale loss and intra scale loss
                #the default self.bata = 1 for only using intra scale loss
                loss_s =  self.beta * loss_intra_scale + (1-self.beta) * loss_inter_sacle

                #weighted sum of style loss and content loss
                loss = self.alpha * loss_s + (1-self.alpha) * loss_c
                # print(loss.item())
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()

            generated_img = stylized_img.detach().cpu()
            generated_img = transforms.functional.to_pil_image(generated_img[0])
            generated_img.save("./data/generate/" + str(epoch) + ".jpg")
                    
            print ("[TRAIN] EPOCH %d/%d, Loss/batch_num: %.4f" % (epoch, MAX_EPOCH, total_loss/(batch_id+1)))
            # torch.save(self.encoder.state_dict(), "./weights/epoch"+str(epoch)+"_encoder.pt")
            # torch.save(self.decoder.state_dict(), "./weights/epoch"+str(epoch)+"_decoder.pt")

model = TotalSytle()
model.train()
