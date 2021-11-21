import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import itertools
import cv2 as cv
import os
import numpy as np
from PIL import Image


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def tensor_to_png(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)

def png_to_tensor(ground_truth_path):
    pil = Image.open(ground_truth_path).convert('RGB')
    #pil = cv.resize(pil, (256,256), interpolation=cv.INTER_CUBIC)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))

class residual_block(nn.Module):
    def __init__(self):
        super(residual_block, self).__init__()
        self.conv_1 = nn.Conv2d(48, 48, 1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(48)

        self.conv_2 = nn.Conv2d(48, 48, 1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(48)
 
    def forward(self, input):
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = F.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        
        return output + input


class DCR(nn.Module):
    def __init__(self):
        super(DCR, self).__init__()
        self.conv_1 = nn.Conv2d(3, 48, 3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(48)

        self.block2 = residual_block()
        self.block3 = residual_block()

        self.conv_4 = nn.Conv2d(48, 48, 3, stride=1, padding=1)

        self.conv_5 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(48)

        self.block6 = residual_block()
        self.block7 = residual_block()

        self.conv_8 = nn.Conv2d(48 ,3, 3, stride=1, padding=1)

        self.conv_9 = nn.Conv2d(3, 48, 3, stride=1, padding=1)
        self.bn_9 = nn.BatchNorm2d(48)

        self.block10 = residual_block()
        self.block11 = residual_block()

        self.conv_12 = nn.Conv2d(48, 48, 3, stride=1, padding=1)

        self.conv_13 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.bn_13 = nn.BatchNorm2d(48)

        self.block14 = residual_block()
        self.block15 = residual_block()

        self.conv_16 = nn.Conv2d(48, 3, 3, stride=1, padding=1)
        
    def forward(self, input):
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = F.relu(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.conv_4(output)
        alpha1 = F.relu(output)

        output = self.conv_5(alpha1)
        output = self.bn_5(output)
        output = F.relu(output)
        output = self.block6(output)
        output = self.block7(output)
        output = self.conv_8(output)
        recons = nn.Sigmoid()(output)

        output = self.conv_9(recons)
        output = self.bn_9(output)
        output = F.relu(output)
        output = self.block10(output)
        output = self.block11(output)
        output = self.conv_12(output)
        alpha2 = F.relu(output)

        output = self.conv_13(alpha2)
        output = self.bn_13(output)
        output = F.relu(output)
        output = self.block14(output)
        output = self.block15(output)
        recyc = self.conv_16(output)



     

        return alpha1, recons, alpha2, recyc




if __name__=='__main__':
    recons_name = './output/'
    im_num = 2
    w_recons = 5
    use_cuda = True

    sigma = 1./30

    num_steps = 5001

    save_frequency = 100
    net = DCR()
    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    save_img_ind = 0
    w_code = 0.00
    for step in range(num_steps):
        loss_main1 = 0
        loss_main2 = 0
        
        for im_index in range(0,1):
            SARground_truth_path = './realt1.png'
            OPTground_truth_path = './realt2.png'
            cloudmask_path = './realmask.png'
        

            SAR = png_to_tensor(SARground_truth_path)
            opt = png_to_tensor(OPTground_truth_path)
            mask = png_to_tensor(cloudmask_path)
            #mask = 1 - mask
            destructedimg = opt * mask
            
            alpha1, recons, alpha2, recyc = net(SAR)
            masked_recons = recons * mask

            optimizer.zero_grad()
            loss_recons = torch.sum(torch.abs(masked_recons - destructedimg))/40000
            loss_recyc = torch.sum(torch.abs(recyc - SAR))/40000
            loss_code = torch.sum(torch.abs(alpha1 - alpha2))/40000
            loss_total = loss_recyc + w_recons * loss_recons + w_code * loss_code
            loss_real = torch.sum(torch.abs(recons - opt))/40000
            loss_main1 = loss_main1 + loss_real
            loss_main2 = loss_main2 + loss_total
            loss_total.backward()
            optimizer.step()
            if step % save_frequency == 0:
                tensor_to_png(recons.data,recons_name+'_{}_{}.png'.format(save_img_ind, im_index))
                #tensor_to_png(destructedimg.data,Cor_name+'_{}.png'.format(im_index))
                #tensor_to_png(opt.data,GT_name+'_{}.png'.format(im_index))
        
        print('At step {}, loss_main1 is {}, loss_main2 is {}'.format(step, loss_main1.data.cpu(), loss_main2.data.cpu()))
        
        if step % save_frequency == 0:
            save_img_ind += 1
    if use_cuda:
        torch.cuda.empty_cache()
