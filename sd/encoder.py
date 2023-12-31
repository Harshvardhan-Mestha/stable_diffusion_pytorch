import torch
from torch import nn
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock,VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1), #(bsize,3,h,w) -> (bsize,128,h,w)

            VAE_ResidualBlock(128,128), #(bsize,128,h,w) -> (bsize,128,h,w)
            VAE_ResidualBlock(128,128), #(bsize,128,h,w) -> (bsize,128,h,w)

            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0), #(bsize,128,h,w) -> (bsize,128,h/2,w/2)

            VAE_ResidualBlock(128,256),#(bsize,128,h/2,w/2) -> (bsize,256,h/2,w/2)
            VAE_ResidualBlock(256,256),#(bsize,256,h/2,w/2) -> (bsize,256,h/2,w/2)

            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),#(bsize,256,h/2,w/2) -> (bsize,256,h/4,w/4)

            VAE_ResidualBlock(256,512),#(bsize,256,h/4,w/4) -> (bsize,512,h/4,w/4)
            VAE_ResidualBlock(512,512),#(bsize,512,h/4,w/4) -> (bsize,512,h/4,w/4)

            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),#(bsize,512,h/4,w/4) -> (bsize,512,h/8,w/8)

            VAE_ResidualBlock(512,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)
            VAE_ResidualBlock(512,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)
            VAE_ResidualBlock(512,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)

            VAE_AttentionBlock(512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)

            VAE_ResidualBlock(512,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)

            nn.GroupNorm(32,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)
            nn.SiLU(),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)

            nn.Conv2d(512,8,kernel_size=3,padding=1),#(bsize,512,h/8,w/8) -> (bsize,8,h/8,w/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0)#(bsize,8,h/8,w/8) -> (bsize,8,h/8,w/8)
        )   
    
    def forward(self, x, noise):

        #x: (bs,in_ch,h,w)
        #noise: (bs,out_ch,h/8,w/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                #(pad_left,pad_right,pad_top,pad_bottom)
                x = F.pad(x,(0,1,0,1))
            x = module(x)

        mean, log_variance = torch.chunk(x,2,dim=1) #(bsize,8,h/8,w/8) -> 2 tensors of size (bsize,4,h/8,w/8)

        log_variance = torch.clamp(log_variance,-30,20) #(bsize,4,h/8,w/8) -> (bsize,4,h/8,w/8)
        variance = log_variance.exp() #(bsize,4,h/8,w/8) -> (bsize,4,h/8,w/8)
        stdev = variance.sqrt() #(bsize,4,h/8,w/8) -> (bsize,4,h/8,w/8)

        x = mean + stdev*noise # N(mean,var) = X = mean + stdev*Z , Z ~ N(0,1)

        x *= 0.18215 #scale o/p with const

        return x
    






