import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)























class conv_block_SE(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_SE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SE_Class(nn.Module):
    def __init__(self, Input_Height, num_filters, ratio = 8):
        super(SE_Class, self).__init__()

        # pool over H dim to get 1*1*C
        self.avg = nn.AvgPool2d(kernel_size=Input_Height)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Non linearity
        self.lin1 = nn.Linear(in_features=num_filters, out_features=num_filters//ratio,
                    bias=False)
        self.lin2 = nn.Linear(in_features=num_filters//ratio, out_features=num_filters,
                    bias=False)
        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.conv_spatial = conv_block_SE(1,1)

    def forward(self, ip):
        x = ip
        x = self.avg(x) # 1 * 64 * 256 * 256 -> 1 * 64 *   1 *   1
        x = x.reshape(x.shape[0],1,1,x.shape[1])  # 1 * 1  * 1 * 64

        x = self.lin1(x)
        x = self.Relu(x)
        x = self.lin2(x)
        x = self.sig(x)

        y = x.reshape(x.shape[0],x.shape[-1],1,1)
        x = y * ip

        x2 = ip * self.conv_spatial(0.5 * ip.mean(dim=1, keepdim=True) + 0.5 * ip.amax(dim=1, keepdim=True))

        return x + x2



class W_SE_Class(nn.Module):
    def __init__(self, Input_Height, num_filters, ratio = 8):
        super(W_SE_Class, self).__init__()

        # pool over H dim to get 1*1*C
        self.avg = nn.AvgPool2d(kernel_size=Input_Height)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Non linearity
        self.lin1 = nn.Linear(in_features=num_filters, out_features=num_filters//ratio,
                    bias=False)
        self.lin2 = nn.Linear(in_features=num_filters//ratio, out_features=num_filters,
                    bias=False)
        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self, ip):
        x = ip
        x = self.avg(x) # 1 * 64 * 256 * 256 -> 1 * 64 *   1 *   1
        x = x.reshape(x.shape[0],1,1,x.shape[1])  # 1 * 1  * 1 * 64

        x = self.lin1(x)
        x = self.Relu(x)
        x = self.lin2(x)
        x = self.sig(x)

        y = x.reshape(x.shape[0],x.shape[-1],1,1)
        x = y * ip


        return x 




























class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,SE_Input_H):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            (nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True),
            (nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.SE = SE_Class(SE_Input_H,ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.SE(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            (nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
























class TwoEnc_SumSeDecModel(nn.Module):
    def __init__(self):
        super(TwoEnc_SumSeDecModel, self).__init__()

        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # High
        self.Conv1_High = conv_block(ch_in=3, ch_out=64,SE_Input_H=256)
        self.Conv2_High = conv_block(ch_in=64, ch_out=128,SE_Input_H=128)
        self.Conv3_High = conv_block(ch_in=128, ch_out=256,SE_Input_H=64)
        self.Conv4_High = conv_block(ch_in=256, ch_out=512,SE_Input_H=32)
        self.Conv5_High = conv_block(ch_in=512, ch_out=1024,SE_Input_H=16)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Art
        self.Conv1_Art = conv_block(ch_in=3, ch_out=64,SE_Input_H=256)
        self.Conv2_Art = conv_block(ch_in=64, ch_out=128,SE_Input_H=128)
        self.Conv3_Art = conv_block(ch_in=128, ch_out=256,SE_Input_H=64)
        self.Conv4_Art = conv_block(ch_in=256, ch_out=512,SE_Input_H=32)
        self.Conv5_Art = conv_block(ch_in=512, ch_out=1024,SE_Input_H=16)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # UP
        self.Up5_Art = up_conv(ch_in=1024, ch_out=512)

        self.Up_conv5_Art = conv_block(ch_in=1024, ch_out=512,SE_Input_H=32)
        self.Up4_Art = up_conv(ch_in=512, ch_out=256)

        self.Up_conv4_Art = conv_block(ch_in=512, ch_out=256,SE_Input_H=64)
        self.Up3_Art = up_conv(ch_in=256, ch_out=128)

        self.Up_conv3_Art = conv_block(ch_in=256, ch_out=128,SE_Input_H=128)
        self.Up2_Art = up_conv(ch_in=128, ch_out=64)
        
        self.Up_conv2_Art = conv_block(ch_in=128, ch_out=64,SE_Input_H=256)
        
        self.Conv_1x1_Art = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #SE
        self.w5_art  = W_SE_Class(16,1024)
        self.w5_hgh  = W_SE_Class(16,1024)
      
        self.w4_art  = W_SE_Class(32,512)
        self.w4_hgh  = W_SE_Class(32,512)
      
        self.w3_art  = W_SE_Class(64,256)
        self.w3_hgh  = W_SE_Class(64,256)
        
        self.w2_art  = W_SE_Class(128,128)
        self.w2_hgh  = W_SE_Class(128,128)
        
        self.w1_art  = W_SE_Class(256,64)
        self.w1_hgh  = W_SE_Class(256,64)



    def forward(self, ART, HGH):

        # High encoding path
        x1_High = self.Conv1_High(HGH)   #1*64*256*256

        x2_High = self.Avgpool(x1_High) #1*64*128*128
        x2_High = self.Conv2_High(x2_High)   #1*128*128*128

        x3_High = self.Avgpool(x2_High) #1*128*64*64
        x3_High = self.Conv3_High(x3_High)  #1*256*64*64

        x4_High = self.Avgpool(x3_High) #1*256*32*32
        x4_High = self.Conv4_High(x4_High) #1*512*32*32

        x5_High = self.Avgpool(x4_High) #1*512*16*16
        x5_High = nn.Tanh()(self.Conv5_High(x5_High))  #1*1024*16*16


        # Art encoding path
        x1_Art = self.Conv1_Art(ART)   #1*64*256*256

        x2_Art = self.Avgpool(x1_Art) #1*64*128*128
        x2_Art = self.Conv2_Art(x2_Art)   #1*128*128*128

        x3_Art = self.Avgpool(x2_Art) #1*128*64*64
        x3_Art = self.Conv3_Art(x3_Art)  #1*256*64*64

        x4_Art = self.Avgpool(x3_Art) #1*256*32*32
        x4_Art = self.Conv4_Art(x4_Art) #1*512*32*32

        x5_Art = self.Avgpool(x4_Art) #1*512*16*16
        x5_Art = nn.Tanh()(self.Conv5_Art(x5_Art)) #1*1024*16*16


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        # decoding + concat path
        x5_Art =  self.w5_art(x5_Art)
        x5_High = self.w5_hgh(x5_High)
        d5_Art = self.Up5_Art(x5_Art+x5_High) #1*512*32*32
        
        x4_Art = self.w4_art(x4_Art)
        x4_High = self.w4_hgh(x4_High)
        d5_Art = torch.cat((x4_Art+x4_High, d5_Art), dim=1) #1*1024*32*32
        d5_Art = self.Up_conv5_Art(d5_Art) ##1*512*32*32
        d4_Art = self.Up4_Art(d5_Art) #1*256*64*64


        x3_Art =  self.w3_art(x3_Art)
        x3_High = self.w3_hgh(x3_High)
        d4_Art = torch.cat((x3_Art+x3_High, d4_Art), dim=1)#1*512*64*64
        d4_Art = self.Up_conv4_Art(d4_Art) #1*256*64*64
        d3_Art = self.Up3_Art(d4_Art) #1*128*128*128


        x2_Art =  self.w2_art(x2_Art)
        x2_High = self.w2_hgh(x2_High)
        d3_Art = torch.cat((x2_Art+x2_High,d3_Art), dim=1) #1*256*128*128
        d3_Art = self.Up_conv3_Art(d3_Art) #1*128*128*128
        d2_Art = self.Up2_Art(d3_Art) #1*64*256*256


        x1_Art =  self.w1_art(x1_Art)
        x1_High = self.w1_hgh(x1_High)
        d2_Art = torch.cat((x1_Art+x1_High, d2_Art), dim=1)#1*128*256*256
        d2_Art = self.Up_conv2_Art(d2_Art)#1*64*256*256
        d1_Art = self.Conv_1x1_Art(d2_Art) #1*1*256*256


        return nn.Tanh()(d1_Art)




def TwoEnc_SumSeDecModel_init():
    model = TwoEnc_SumSeDecModel()
    init_weights(model, 'normal')
    return model
