import torch

def absmax_quantize(X):
    scale = 127 / X.abs().max()
    x_quant = (scale * X).round()
    x_dequant = x_quant / scale

    return x_quant.to(torch.int8), x_dequant

def zeropoint_quantize(X):
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range 

    scale = 255 / x_range

    zerpoint = ((-scale * torch.min(X)) - 128).round()
    
    x_quant = (scale * X + zerpoint).round()
    x_dequant = (x_quant - zerpoint) / scale

    return x_quant.to(torch.uint8), x_dequant