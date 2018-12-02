import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
import scipy.misc
from torch.utils.serialization import load_lua
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import Encoder, Decoder, StyleLoss, ContentLoss
from utils import MST


MAX_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def load_data():
    '''
    ## TODO: return (style_imgs, content_imgs)

    style_imgs: Nx3x224x224
    content_imgs: Nx3x224x224

    This funciton is called in TotalStyle().__inti__() to load the data
    '''
    return style_imgs, content_imgs

class MyCostumeDataset(Dataset):
    def __init__(self, style, content):
        self.train_s = style
        self.train_c = content

    def __getitem__(self, index):
        style = self.train_s[index]
        content = self.train_c[index]
        return torch.tensor(style), torch.tensor(content)

    def __len__(self):
        return len(self.train_s)

class TotalSytle():
    def __init__(self):
        style_imgs, content_imgs = load_data() ## TODO: load the style and content pair
        dataset = MyCostumeDataset(style=style_imgs, content=content_imgs)
        self.train_loader = DataLoader(dataset, batch_size=BATCH_SIZE , shuffle=True)
        print ("----------------------Data is loaded----------------------------")
        print ("Training Dataset: ", len(dataset))

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.style_loss = StyleLoss() ## TODO: Complete Styleloss
        self.content_loss = ContentLoss() ## TODO: Complete ContentLoss
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

                #compute the loss between stylized imgs and style imgs
                loss_s = self.style_loss(encoded_stylized, encoded_style)
                ## TODO: Complete style loss

                #compute the loss between stylized imgs and content imgs
                #use only relu3_3 to as the 'content' of an img
                loss_c = self.content_loss(encoded_stylized[-1], relu3_3)
                ## TODO: Complete content loss
                #weighted sum of style loss and content loss
                loss = self.alpha * loss_s + (1-self.alpha) * loss_c
                loss.backward()
                self.optimizer.step()
        print ("[TRAIN] EPOCH %d/%d, Loss/batch_num: %.4f" % (epoch, MAX_EPOCH, loss.item()/batch_id))
        torch.save(self.encoder.state_dict(), "./weights/epoch"+str(epoch)+"_encoder.pt")
        torch.save(self.decoder.state_dict(), "./weights/epoch"+str(epoch)+"_decoder.pt")
