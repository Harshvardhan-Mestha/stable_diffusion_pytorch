import torch 
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention
import math

class TimeEmbedding(nn.Module):
    def __init__(self,n_embd:int):
        super().__init__()

        self.linear_1 = nn.Linear(n_embd,4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd,4*n_embd)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #x(1,320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x #(1,1280)
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n_time=1280):
        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time,out_channels)
        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self,feature,time):
        #feature: (bsize,in_ch,h,w)
        #time:(1,1280)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int, n_embd:int, d_context=768) -> None:
        super().__init__()

        channels = n_head*n_embd

        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels,channels, kernel_size=1,padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head,channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head,channels,d_context,in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)
        self.conv_output = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        # x: (bsize,ch,h,w)
        # context: (bsize, seqlen, dim)

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n,c,h,w = x.shape

        x = x.view(n,c,h*w) #(bsize,ch,h,w) -> (bsize,ch,h*w)
        x = x.transpose(-1,-2) #(bsize,ch,h*w) -> (bsize,h*w,ch)

        #norm + self attention + skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x,False)
        x += residue_short
        residue_short = x
        #norm + cross attention + skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x,context)
        x += residue_short
        residue_short = x

        # norm + Feed Fwd + GeGLU + skip connection

        x = self.layernorm_3(x)
        x,gate = self.linear_geglu_1(x).chunk(2,dim = -1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))

        return self.conv_output(x) + residue_long







class Upsample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x:torch.Tensor):
        x = F.interpolate(x,scale_factor=2,mode='nearest') #(bs,c,h,w) -> (bs,c,h*2,w*2)
        return self.conv(x)
    
class SwitchSequential(nn.Sequential):
    def forward(self,x:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:

        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x,context)
            elif isinstance(layer,UNET_ResidualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            #(bsize,4,h/8,w/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            #(bsize,320,h/8,w/8) -> (bsize,320,h/16,w/16)
            SwitchSequential(nn.Conv2d(320,320, kernel_size=3,stride=2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),

            #(bsize,640,h/16,w/16) -> (bsize,640,h/32,w/32)
            SwitchSequential(nn.Conv2d(640,640, kernel_size=3,stride=2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),

            #(bsize,1280,h/32,w/32) -> (bsize,1280,h/64,w/64)
            SwitchSequential(nn.Conv2d(1280,1280, kernel_size=3,stride=2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

            #(bsize,1280,h/64,w/64) -> (bsize,1280,h/64,w/64)
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280),
        )

        self.decoders = nn.ModuleList([
            #(bsize,2560,h/64,w/64) -> (bsize,1280,h/64,w/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8,160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8,80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8,40)),
            
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40)),
    
        ])
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()

        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x:torch.Tensor):
        # x: (bs,320,h/8,w/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x # x: (bs,4,h/8,w/8)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()


        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor)-> torch.Tensor:

        #latent (bsize,4,h/8,w/8) (encoder.py output dim)
        #context (bsize,seqlen,dim) (clip.py output dim)
        #time (1,320) (time embedding dim)

        time = self.time_embedding(time) #(1,320) -> (1,1280)
        output = self.unet(latent,context,time) #(bsize,4,h/8,w/8) -> (bsize,320,h/8,w/8)
        output = self.final(output) #(bsize,320,h/8,w/8) -> (bsize,4,h/8,w/8)
        
        return output #(bsize,4,h/8,w/8)






