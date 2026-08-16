import os
import torch
import numpy as np
import math

data_dir = os.path.join('dataset', 'machado')
device = 'cuda'
block_size = 256

def perplexity(model):
    data = np.memmap(
        os.path.join(data_dir, 'val.bin'),
        dtype=np.uint16,
        mode='r'
    )

    n = (len(data) - 1) // block_size

    starts = list(range(0, n*block_size, block_size))
    total = 0.0
    count = 0

    with torch.no_grad():
        model.eval()
        model.to(device)

        for b in range(0, len(starts), 128):
            chunk = starts[b:b+128]

            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in chunk]).to(device)
            y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in chunk]).to(device)


        _, loss, _ = model(x, targets=y)

        total += loss.item() * x.size(0)
        count += x.size(0)

    return math.exp(total/count)
