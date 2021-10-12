#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def img_to_spatial(x,kernel_size=None):
    bs,channels,img_size,_ = x.shape
    out = torch.zeros(bs,channels,img_size,img_size)
    for batch in range(bs):
        for k in range(channels):
            out[batch][k] = torch.fft.ifft2(x[batch][k]).real#[:,:,0]
    if not kernel_size is None:
        out = out[:,:,0:-kernel_size+1]
        out = out[:,:,:,0:-kernel_size+1]
        
    return out

def img_to_freq(x):
    bs,channels,img_size,_ = x.shape
    out = torch.zeros((bs,channels,img_size,img_size), dtype=torch.cfloat)
    x = x.reshape(bs,channels,img_size,img_size)
    for batch in range(bs):
        for k in range(channels):
            out[batch][k] = torch.fft.fft2(x[batch][k])
    return out

def get_mean_var(sx):
    bs,channels,_,_ = sx.shape            
    sx = sx.reshape(bs,channels,-1)
    sx = sx.permute(1,0,2).reshape(3,-1)
            
    mean = torch.mean(sx,dim = -1)
    var = torch.mean((sx-mean.reshape(-1,1))**2,dim = -1)
    return mean,var

def get_mean_var_fourier(x):
    sx = img_to_spatial(x)
    return get_mean_var(sx)

class FBatchnorm2d(nn.Module):
    """
    Batchnormalization:
    https://arxiv.org/pdf/1502.03167.pdf
    This version of Batchnorm is highly influenct by fastai's running batchnorm

    Performs Normalisation ChannelWise!
    Each Channel is seperatly a frequence domain
    """
    def __init__(self,channels,momentum = 0.8,epsilon=1e-5):
        super(FBatchnorm2d, self).__init__()
        self.running_mean = torch.zeros(channels,requires_grad = False)
        self.running_var = torch.ones(channels,requires_grad = False)
        self.momentum,self.epsilon = momentum,epsilon
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        
        #this says https://arxiv.org/abs/1706.02677 do
    def _normalize(self,x):
        for k in range(self.running_mean.shape[0]):
            x[:,k] = self.gamma[k] * ((x[:,k] - self.running_mean[k]) / torch.sqrt(self.running_var[k] + self.epsilon)) + self.beta[k]
        return x

    def forward(self,x):
        bs,channels,_,_ = x.shape            
        if self.training:
            mean,var = get_mean_var_fourier(x)
            self.running_mean = self.running_mean * self.momentum + (1-self.momentum) * mean
            self.running_var = self.running_var * self.momentum + (1-self.momentum) * var
        return self._normalize(x)

def fconv2d(fweights,fx):
    """
    fx is viewed as complex!
    each channel is seperatly in frequence domain.
    so fx is actually a stack of frequence domains.
    """
    out_channels,in_channels,kernel_size,_ = tuple(fweights.shape)
    bs,img_size = fx.shape[0],fx.shape[-1]
    out = torch.zeros((bs,out_channels,img_size,img_size), dtype=torch.cfloat)
    for i in range(out_channels):
        #TODO write as Matrix Multiply
        for j in range(in_channels):
            out[:,i] = out[:,i] + fweights[i,j] * fx[:,j]        
    return out

def construct_shift_vec(N,shift):
    return torch.tensor([math.e **(-2j*math.pi*k * (shift) / N) for k in range(N)])

def construct_shift_matrix(N,y_shift,x_shift):
    out = [[ math.e **(-2j*math.pi*(n1*x_shift + n2*y_shift)/N) for n1 in range(N) ] for n2 in range(N)]
    return torch.tensor(out)


class FConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,shift_output):
        """
            TODO implement stride
        """
        super(FConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels,self.out_channels = in_channels,out_channels
        self.weights = torch.nn.Parameter(torch.zeros(out_channels,in_channels,kernel_size,kernel_size))
        torch.nn.init.xavier_uniform_(self.weights)
        if self.kernel_size > 5:
            raise RuntimeError("fourier conv is to inacurate!")
        self.shift_output = shift_output
    def forward_correct(self,x):
        #x = F.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value = 0)
        return F.conv2d(x,self.weights)
        
    def get_fweights(self,img_size):
        if (img_size - self.kernel_size) % 2 != 0:
            raise RuntimeError("img_size - self.kernel_size not devidable with 2")
        pa = (img_size - self.kernel_size)
        W = nn.functional.pad(input=self.weights, pad=(0,pa,0,pa), mode='constant', value = 0)
        fweights = torch.conj(torch.fft.fft2(W))
        """
        we multiply by construct_shift_matrix(img_size,self.kernel_size//2,self.kernel_size//2)
        to make center the output and implicitly pad
        with F.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value = 0)
        """
        if self.shift_output:
            return fweights * construct_shift_matrix(img_size,self.kernel_size//2,self.kernel_size//2)
        else:
            return fweights
    def forward(self,x):
        """
        x is viewed as complex number
        """
        fweights = self.get_fweights(x.shape[-1])
        out =  fconv2d(fweights,x)
        return out
        


class modReLU(nn.Module):
    """
    implementation is based on https://arxiv.org/pdf/1705.09792.pdf
    """
    def __init__(self):
        super(modReLU, self).__init__()