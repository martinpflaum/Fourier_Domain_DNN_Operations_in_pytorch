# Fourier_Domain_DNN_Operations_in_pytorch
Some Fourier Domain DNN Operations.
The FConv2d takes as input an image that is in the fourier/frequency domain. img_to_spatial takes as input an image that is in the fourier domain and converts it to an image that is in the spatial domain. img_to_freq takes as input an image, that is in the spatial domain and converts it to an image that is in the frequency domain.
shift_ouput=True will shift the output into the middle of the 2D image tensor. It will only work with odd kernel_sized tensors. By default the result will apear at the top left. img_to_spatial by default doesn't cut off values that do not belong to the result. If you want that behaviour you will need to set kernel_size to the the kernel_size of the convolution operation.
```python
import torch
from operations import FConv2d,FBatchnorm2d,img_to_spatial,\
img_to_freq,get_mean_var_fourier,get_mean_var

conv = FConv2d(3,1,3,shift_output = False)
x = torch.randn(1,3,5,5)
x1 = x
print(img_to_spatial(conv(img_to_freq(x)),kernel_size=3))
print(conv.forward_correct(x1))
```
With shift_output = True the output is actually identically as if you would pad the input circular.
```python
conv = FConv2d(3,1,5,shift_output = True)
x = torch.randn(1,3,5,5)
x1 = torch.nn.functional.pad(input=x, pad=(2, 2, 2, 2), mode='circular')

print(img_to_spatial(conv(img_to_freq(x))))
print(conv.forward_correct(x1))
```
FBatchnorm2d operates also in the fourier domain, but internally converts some values into the spatial domain durring training.
```python
batchnorm = FBatchnorm2d(channels=3)
xorg = torch.randn(4,3,5,5)
x = img_to_freq(xorg)
y = batchnorm(x)

print(get_mean_var_fourier(y))
print(get_mean_var(xorg))
```