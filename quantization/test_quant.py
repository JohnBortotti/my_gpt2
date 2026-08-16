import torch
from models.nanogpt import GPT, Config
from quantization.absmax_zeropoint import absmax_quantize, absmax_quantize_4bits, zeropoint_quantize
from quantization.perplexity import perplexity
import numpy as np
from copy import deepcopy
import os
import pickle
from contextlib import nullcontext

device = 'cuda'
ctx = nullcontext()

checkpoint = torch.load('outputs/machado-char/ckpt.pt', map_location=device)

config = Config(**checkpoint['model_args'])
model = GPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

weights = [param.data.clone() for param in model.parameters()]

# absmax 8 bits
model_abs = deepcopy(model)
weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param)
    param.data = dequantized
    weights_abs.append(dequantized)

# zeropoint 8 bits
model_zp = deepcopy(model)
weights_zp = []
for param in model_zp.parameters():
    _, dequantized = zeropoint_quantize(param)
    param.data = dequantized
    weights_zp.append(dequantized)

# absmax 4 bits
model_abs4 = deepcopy(model)
weights_abs = []
for param in model_abs4.parameters():
    _, dequantized = absmax_quantize_4bits(param)
    param.data = dequantized
    weights_abs.append(dequantized)

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

# print("Full precision model:")
# generate_text(model)
#
# print("\nAbsmax quant model:")
# generate_text(model_abs)
#
# print("\nZeropoint quant model:")
# generate_text(model_zp)
#
# print("\nAbsmax (4 bits) quant model:")
# generate_text(model_abs4)


# print("\nPerplexity:")
print(f"Full precision model: {perplexity(model)}")
print(f"Absmax: {perplexity(model_abs)}")
print(f"ZeroPoint: {perplexity(model_zp)}")
print(f"Absmax (4 bits): {perplexity(model_abs4)}")
