import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Loader import Dataset
import scipy.misc
from torch.utils.serialization import load_lua
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import Encoder, Decoder, StyleLoss, ContentLoss
from img_loader import get_data_loader


MAX_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CONTENT_PATH = "./data/content_img/",
STYLE_PATH = "./data/style_img/",

def load_data():
    '''
    ## TODO: return (style_imgs, content_imgs)

    style_imgs: Nx3x224x224
    content_imgs: Nx3x224x224

    This funciton is called in TotalStyle().__inti__() to load the data
    '''
    return style_imgs, content_imgs

def calc_mean_cov(cF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentCov = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)
    
    c_d = (c_e).pow(-0.5)
    temp = torch.mm(c_v,torch.diag(c_d))
    c_cov = torch.mm(temp,(c_v.t()))

    return c_mean, c_cov

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
        print ("----------------------Data is loaded----------------------------")
        print ("Training Dataset: ", len(dataset))

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mse_loss = torch.nn.MSELoss()
        # self.style_loss = StyleLoss() ## TODO: Complete Styleloss
        # self.content_loss = ContentLoss() ## TODO: Complete ContentLoss
        print ("----------------------Model is loaded----------------------------")
        parameters = list(self.encoder.parameters())+list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

        self.use_gpu = True
        if self.use_gpu:
            print ("----------------------GPU is used to train---------------------------")
        else:
            print ("----------------------CPU is used to train----------------------------")

        self.alpha = 0.5

    def train(self):
        total_loss = 0
        for epoch in range(MAX_EPOCH):
            for batch_id, (style_imgs, content_imgs) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                #Parse the style_imgs and content_imgs into encoder
                encoded_style, output_style = self.encoder(sytle_imgs)
                encoded_conent, output_content = self.encoder(content_imgs)

                #Compute the MST transformed relu
                relu1_2, relu2_2, relu3_3 = MST(encoded_styles, encoded_conent)

                #Skip connection with decoder
                stylized_img = self.decoder(relu1_2, relu2_2, relu3_3)

                #Extract the features of generated stylized img
                encoded_stylized, _ = self.encoder(stylized_img)

                #compute the loss between stylized imgs and content imgs
                #use only relu3_3 to as the 'content' of an img
                loss_c = self.mse_loss(encoded_stylized[-1], relu3_3)

                #compute the loss between stylized imgs and style imgs
                # intra scale loss
                loss_s = calc_style_loss(encoded_stylized[0], encoded_style[0], mse_loss)
                for i in range(1, 3):
                    loss_s += calc_style_loss(encoded_stylized[i], encoded_style[i], mse_loss)

                # inter scale loss
                encoded_stylized = upsample_and_cat(encoded_stylized)
                encoded_content = upsample_and_cat(encoded_content)

                loss_s += calc_style_loss(encoded_stylized, encoded_content, mse_loss)

                #weighted sum of style loss and content loss
                loss = self.alpha * loss_s + (1-self.alpha) * loss_c
                loss.backward()
                self.optimizer.step()
        print ("[TRAIN] EPOCH %d/%d, Loss/batch_num: %.4f" % (epoch, MAX_EPOCH, loss.item()/batch_id))
        torch.save(self.encoder.state_dict(), "./weights/epoch"+str(epoch)+"_encoder.pt")
        torch.save(self.decoder.state_dict(), "./weights/epoch"+str(epoch)+"_decoder.pt")
