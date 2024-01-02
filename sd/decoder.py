import torch 
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        residue = x 
        x = self.groupnorm(x)
        n,c,h,w = x.shape

        x = x.view(n,c,h*w) #(n,c,h*w)
        x = x.transpose(-1,-2) #(n,h*w,c)
        x = self.attention(x,causal_mask=False)
        x = x.transpose(-1,-2) #(n,c,h*w)
        x = x.view(n,c,h,w)
        x += residue

        return x
        
        



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

        self.groupnorm_2 = nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #x: (bs,in_ch,h,w)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),#(bsize,512,h/8,w/8) -> (bsize,512,h/8,w/8)
            nn.Upsample(scale_factor=2),#(bsize,512,h/8,w/8) -> (bsize,512,h/4,w/4)
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.Upsample(scale_factor=2),#(bsize,512,h/4,w/4) -> (bsize,512,h/2,w/2)
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            nn.Upsample(scale_factor=2),#(bsize,256,h/2,w/2) -> (bsize,256,h,w)
            nn.Conv2d(256,256,kernel_size=3,padding=1),

            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128),

            nn.SiLU(),

            nn.Conv2d(128,3,kernel_size=3,padding=1)#(bsize,128,h/2,w/2) -> (bsize,3,h,w)
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x:(bsize,4,h/8,w/8)

        x /= 0.18125

        for module in self:
            x = module(x)

        return x #(bsize,3,h,w)