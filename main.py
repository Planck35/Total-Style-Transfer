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

def calc_mean_cov(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    N, C = size[:2]
    # feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_cov = cov(feat.view(N, C, -1)).view(N, C, 1, 1)
    return feat_mean, feat_cov

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def calc_style_loss(input, target, loss_fn):
    m1, s1 = calc_mean_cov(encoded_stylized)
    m2, s2 = calc_mean_cov(encoded_style)
    return loss_fn(m1, m2) + loss_fn(s1, s2)

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
                loss_s = calc_style_loss(encoded_stylized[0], encoded_style[0])
                for i in range(1, 4): # ??? Huang
                    loss_s += calc_style_loss(encoded_stylized[i], encoded_style[i])
                # inter scale loss
                # TODO: complete inter scale loss


                #weighted sum of style loss and content loss
                loss = self.alpha * loss_s + (1-self.alpha) * loss_c
                loss.backward()
                self.optimizer.step()
        print ("[TRAIN] EPOCH %d/%d, Loss/batch_num: %.4f" % (epoch, MAX_EPOCH, loss.item()/batch_id))
        torch.save(self.encoder.state_dict(), "./weights/epoch"+str(epoch)+"_encoder.pt")
        torch.save(self.decoder.state_dict(), "./weights/epoch"+str(epoch)+"_decoder.pt")
