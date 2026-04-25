import torch
from models.nanogpt import GPT, Config
from quantization.absmax_zeropoint import absmax_quantize, zeropoint_quantize
import numpy as np
from copy import deepcopy
import os
import pickle
from contextlib import nullcontext

device = 'mps'
ctx = nullcontext()

checkpoint = torch.load('outputs/machado-char/ckpt.pt')

config = Config(**checkpoint['model_args'])
model = GPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

weights = [param.data.clone() for param in model.parameters()]

model_abs = deepcopy(model)
weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param)
    param.data = dequantized
    weights_abs.append(dequantized)

model_zp = deepcopy(model)
weights_zp = []
for param in model_zp.parameters():
    _, dequantized = zeropoint_quantize(param)
    param.data = dequantized
    weights_zp.append(dequantized)

def generate_text(model):
    input_text = "bom dia"
    model.eval()
    model.to(device)
    
    meta_path = os.path.join('dataset', 'machado', 'meta.pkl')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    start_ids = encode(input_text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device) [None, ...])

    with torch.no_grad():
        with ctx:
            for k in range(1):
                y = model.generate(x, 500, temperature=0.9, top_k=5, use_kv_cache=False)
                print(decode(y[0].tolist()))
                print('-------------')

print("Full precision model:")
generate_text(model)

print("\nAbsmax quant model:")
generate_text(model_abs)

print("\nZeropoint quant model:")
generate_text(model_zp)