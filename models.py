import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self,in_filters,out_filters,seprable=True):
        super(conv_block,self).__init__()

        if seprable:
            self.spatial1 = nn.Conv2d(in_filters,in_filters,kernel_size=3,groups=in_filters,padding=1)
            self.depth1 = nn.Conv2d(in_filters,out_filters,kernel_size=1)

            self.conv1=lambda x: self.depth1(self.spatial1(x))

            self.spatial2 = nn.Conv2d(out_filters,out_filters,kernel_size=3,groups=out_filters,padding=1) 
            self.depth2 = nn.Conv2d(out_filters,out_filters,kernel_size=1)

            self.conv2 = lambda x:self.depth2(self.spatial2(x))
        else:
            self.conv1 = nn.Conv2d(in_filters,out_filters,kernel_size=3,padding=1)
            self.conv2 = nn.Conv2d(out_filters,out_filters,kernel_size=3,stride=1,padding=1)

        self.batchnorm1 = nn.BatchNorm2d(out_filters)
        self.batchnorm2 = nn.BatchNorm2d(out_filters)

    def forward(self,x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))

        return x

class up_conv(nn.Module):
    def __init__(self,in_filter,out_filters):
        super(up_conv,self).__init__()

        self.up = nn.ConvTranspose2d(in_filter,out_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.batchnorm = nn.BatchNorm2d(out_filters)

    def forward(self,x):
        x = F.relu(self.batchnorm(self.up(x)))
        return x
    

class Recurrent_block(nn.Module):
    def __init__(self,out_filters,t=2,seprable=True):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = out_filters

        if seprable:
            self.spatial = nn.Conv2d(self.ch_out,self.ch_out,kernel_size=3,stride=1,groups=self.ch_out,padding=1)
            self.depth = nn.Conv2d(self.ch_out,self.ch_out,kernel_size=1)
            self.conv = lambda x:self.depth(self.spatial(x))
        
        else:
            self.conv = nn.Conv2d(self.ch_out,self.ch_out,kernel_size=3,stride=1,padding=1)
        
        self.batchnorm = nn.BatchNorm2d(self.ch_out)

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = F.relu(self.batchnorm(self.conv(x)))

            x1 = F.relu(self.batchnorm(self.conv(x+x1)))
        
        return x1
    
class RCNN_block(nn.Module):
    def __init__(self,in_filter,out_filters,t=2,seprable=True):
        super(RCNN_block,self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_filters,t=t,seprable=seprable),
            Recurrent_block(out_filters,t=t,seprable=seprable)
        )

        self.Conv_1x1 = nn.Conv2d(in_filter,out_filters,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,kernel_size=1,stride=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,F_int,kernel_size=1,stride=1),
            nn.BatchNorm2d(F_int) 
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int,1,kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(True)
    
    def forward(self,g,x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,ch_mul=64):
        super(U_Net,self).__init__()

        self.Conv1 = conv_block(img_ch,ch_mul,seprable=False)
        self.Conv2 = conv_block(ch_mul,ch_mul*2)
        self.Conv3 = conv_block(ch_mul*2,ch_mul*4)
        self.Conv4 = conv_block(ch_mul*4,ch_mul*8)
        self.Conv5 = conv_block(ch_mul*8,ch_mul*16)

        self.Up5 = up_conv(ch_mul*16,ch_mul*8)
        self.Up_conv5 = conv_block(ch_mul*16,ch_mul*8)

        self.Up4 = up_conv(ch_mul*8,ch_mul*4)
        self.Up_conv4 = conv_block(ch_mul*8,ch_mul*4)

        self.Up3 = up_conv(ch_mul*4,ch_mul*2)
        self.Up_conv3 = conv_block(ch_mul*4,ch_mul*2)

        self.Up2 = up_conv(ch_mul*2,ch_mul)
        self.Up_conv2 = conv_block(ch_mul*2,ch_mul,seprable=False)

        self.Conv_1x1 = nn.Conv2d(ch_mul,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        #encode
        x1 = self.Conv1(x)

        x2 = self.Conv2(F.max_pool2d(x1,(2,2)))

        x3 = self.Conv3(F.max_pool2d(x2,(2,2)))

        x4 = self.Conv4(F.max_pool2d(x3,(2,2)))

        x5 = self.Conv5(F.max_pool2d(x4,(2,2)))

        # decode + skipconnection

        d5 = torch.cat([x4,self.Up5(x5)],1)
        d5 = self.Up_conv5(d5)

        d4 = torch.cat([x3,self.Up4(d5)],1)
        d4 = self.Up_conv4(d4)

        d3 = torch.cat([x2,self.Up3(d4)],1)
        d3 = self.Up_conv3(d3)

        d2 = torch.cat([x1,self.Up2(d3)],1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2,ch_mul=64):
        super(R2U_Net,self).__init__()

        self.R2CNN1 = RCNN_block(img_ch,out_filters=ch_mul,t=t,seprable=False)
        self.R2CNN2 = RCNN_block(ch_mul,ch_mul*2,t=t)
        self.R2CNN3 = RCNN_block(ch_mul*2,ch_mul*4,t=t)
        self.R2CNN4 = RCNN_block(ch_mul*4,ch_mul*8,t=t)
        self.R2CNN5 = RCNN_block(ch_mul*8,ch_mul*16,t=t)

        self.Up5 = up_conv(ch_mul*16,ch_mul*8)
        self.Up_R2CNN5 = RCNN_block(ch_mul*16,ch_mul*8,t=t)
        
        self.Up4 = up_conv(ch_mul*8,ch_mul*4)
        self.Up_R2CNN4 = RCNN_block(ch_mul*8,ch_mul*4,t=t)

        self.Up3 = up_conv(ch_mul*4,ch_mul*2)
        self.Up_R2CNN3 = RCNN_block(ch_mul*4,ch_mul*2,t=t)

        self.Up2 = up_conv(ch_mul*2,ch_mul)
        self.Up_R2CNN2 = RCNN_block(ch_mul*2,ch_mul,seprable=False)

        self.Conv_1x1 = nn.Conv2d(ch_mul,output_ch,kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #encoding
        x1 = self.R2CNN1(x)

        x2 = self.R2CNN2(F.max_pool2d(x1,(2,2)))

        x3 = self.R2CNN3(F.max_pool2d(x2,(2,2)))

        x4 = self.R2CNN4(F.max_pool2d(x3,(2,2)))

        x5 = self.R2CNN5(F.max_pool2d(x4,(2,2)))

        #decoding+ concat

        d5 = torch.cat([x4,self.Up5(x5)],1)
        d5 = self.Up_R2CNN5(d5)

        d4 = torch.cat([x3,self.Up4(d5)],1)
        d4 = self.Up_R2CNN4(d4)

        d3 = torch.cat([x2,self.Up3(d4)],1)
        d3 = self.Up_R2CNN3(d3)

        d2 = torch.cat([x1,self.Up2(d3)],1)
        d2 = self.Up_R2CNN2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)

class AttenU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,ch_mul=64):
        super(AttenU_Net,self).__init__()

        self.Conv1 = conv_block(img_ch,ch_mul,seprable=False)
        self.Conv2 = conv_block(ch_mul,ch_mul*2)
        self.Conv3 = conv_block(ch_mul*2,ch_mul*4)
        self.Conv4 = conv_block(ch_mul*4,ch_mul*8)
        self.Conv5 = conv_block(ch_mul*8,ch_mul*16)

        self.Up5 = up_conv(ch_mul*16,ch_mul*8)
        self.Attn5 = Attention_block(F_g=ch_mul*8,F_l=ch_mul*8,F_int=ch_mul*4)
        self.Up_conv5 = conv_block(ch_mul*16,ch_mul*8)

        self.Up4 = up_conv(ch_mul*8,ch_mul*4)
        self.Attn4 = Attention_block(F_g=ch_mul*4,F_l=ch_mul*4,F_int=ch_mul*2)
        self.Up_conv4 = conv_block(ch_mul*8,ch_mul*4)

        self.Up3 = up_conv(ch_mul*4,ch_mul*2)
        self.Attn3 = Attention_block(F_g=ch_mul*2,F_l=ch_mul*2,F_int=ch_mul)
        self.Up_conv3 = conv_block(ch_mul*4,ch_mul*2)

        self.Up2 = up_conv(ch_mul*2,ch_mul)
        self.Attn2 = Attention_block(F_g=ch_mul,F_l=ch_mul,F_int=ch_mul//2)
        self.Up_conv2 = conv_block(ch_mul*2,ch_mul,seprable=False)
        

        self.Conv_1x1 = nn.Conv2d(ch_mul,output_ch,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #encoding

        x1 = self.Conv1(x)

        x2 = self.Conv2(F.max_pool2d(x1,(2,2)))

        x3 = self.Conv3(F.max_pool2d(x2,(2,2)))

        x4 = self.Conv4(F.max_pool2d(x3,(2,2)))

        x5 = self.Conv5(F.max_pool2d(x4,(2,2)))

        #decoding + concat

        d5 = self.Up5(x5)
        x4 = self.Attn5(d5,x4)
        d5 = torch.cat([x4,d5],1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Attn4(d4,x3)
        d4 = torch.cat([x3,d4],1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Attn3(d3,x2)
        d3 = torch.cat([x2,d3],1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Attn2(d2,x1)
        d2 = torch.cat([x1,d2],1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)

class R2AttenU_Net(nn.Module):
    def __init__(self,img_ch=1,out_ch=1,t=2,ch_mul=64):
        super(R2AttenU_Net,self).__init__()

        self.R2CNN1 = RCNN_block(img_ch,ch_mul,t=t,seprable=False)

        self.R2CNN2 = RCNN_block(ch_mul,ch_mul*2,t=t)
        self.R2CNN3 = RCNN_block(ch_mul*2,ch_mul*4,t=t)
        self.R2CNN4 = RCNN_block(ch_mul*4,ch_mul*8,t=t)
        self.R2CNN5 = RCNN_block(ch_mul*8,ch_mul*16,t=t)

        self.Up5 = up_conv(ch_mul*16,ch_mul*8)
        self.Attn5 = Attention_block(F_g=ch_mul*8,F_l=ch_mul*8,F_int=ch_mul*4)
        self.Up_R2CNN5 = RCNN_block(ch_mul*16,ch_mul*8,t=t)

        self.Up4 = up_conv(ch_mul*8,ch_mul*4)
        self.Attn4 = Attention_block(F_g=ch_mul*4,F_l=ch_mul*4,F_int=ch_mul*2)
        self.Up_R2CNN4 = RCNN_block(ch_mul*8,ch_mul*4,t=t)

        self.Up3 = up_conv(ch_mul*4,ch_mul*2)
        self.Attn3 = Attention_block(F_g=ch_mul*2,F_l=ch_mul*2,F_int=ch_mul)
        self.Up_R2CNN3 = RCNN_block(ch_mul*4,ch_mul*2,t=t)

        self.Up2 = up_conv(ch_mul*2,ch_mul)
        self.Attn2 = Attention_block(F_g=ch_mul,F_l=ch_mul,F_int=ch_mul//2)
        self.Up_R2CNN2 = RCNN_block(ch_mul*2,ch_mul,t=t,seprable=False)

        self.Conv_1x1 = nn.Conv2d(ch_mul,out_ch,kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #encoding 

        x1 = self.R2CNN1(x)

        x2 = self.R2CNN2(F.max_pool2d(x1,(2,2)))

        x3 = self.R2CNN3(F.max_pool2d(x2,(2,2)))

        x4 = self.R2CNN4(F.max_pool2d(x3,(2,2)))

        x5 = self.R2CNN5(F.max_pool2d(x4,(2,2)))
   

        #decoding + concat
        d5 = self.Up5(x5)
        x4 = self.Attn5(d5,x4)
        d5 = torch.cat([x4,d5],1)
        d5 = self.Up_R2CNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Attn4(d4,x3)
        d4 = torch.cat([x3,d4],1)
        d4 = self.Up_R2CNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Attn3(d3,x2)
        d3 = torch.cat([x2,d3],1)
        d3 = self.Up_R2CNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Attn2(d2,x1)
        d2 = torch.cat([x1,d2],1)
        d2 = self.Up_R2CNN2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)
















