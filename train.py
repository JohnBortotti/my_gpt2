import os 
import numpy as np
import torch
from contextlib import nullcontext
import pickle
from models.nanogpt import Config, GPT
import math
import time

eval_only = False

gradient_accumulation_steps = 1
block_size = 256

learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
lr_decay_iters = 2000
min_lr = 1e-4

warmup_iters = 100

device = 'mps'
device_type = 'cpu'
dtype = 'float16'
ctx = nullcontext()
compile = False

data_dir = os.path.join('dataset', 'machado')

def get_batch(split):
    batch_size = 64

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model):
    eval_iters = 50

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, kv_cache = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train():
    iter_num = 0
    best_val_loss = 1e9

    eval_interval = 250

    model_args = dict(
        n_layer=6, n_head=6, n_embd=384, block_size=block_size,
        bias=False, vocab_size=None, dropout=0.2
    )

    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']

    model_args['vocab_size'] = meta_vocab_size

    gptconfig = Config(**model_args)
    model = GPT(gptconfig)

    model.to(device)

    t0 = time.time()
    local_iter_num = 0
    raw_model = model
    decay_lr = True

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    config = {k: globals()[k] for k in config_keys}

    X, Y = get_batch('train')

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    torch.save(checkpoint, os.path.join('outputs/machado-char', 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break
    
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss, kv_cache = model(X, Y)
                loss = loss / gradient_accumulation_steps

            X, Y = get_batch('train')

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        log_interval = 10
        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
        iter_num += 1
        local_iter_num += 1

        max_iters = 5000
        if iter_num > max_iters:
            break

train()