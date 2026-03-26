import torch
from contextlib import nullcontext
import os
from model import Config, GPT
import pickle

out_dir = "outputs/machado-char"
start = "Meu caro amigo "
num_samples = 2
max_new_tokens = 500
temperature = 0.9
top_k = 5
device = 'mps'
device_type = 'cpu'
dtype = 'float16'
use_kv_cache = True
ctx = nullcontext()

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = Config(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

meta_path = os.path.join('dataset', 'machado', 'meta.pkl')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device) [None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, use_kv_cache=use_kv_cache)
            print(decode(y[0].tolist()))
            print('-------------')