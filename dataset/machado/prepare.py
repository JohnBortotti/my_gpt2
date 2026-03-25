import os
import pickle
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), 'inputs')
data = ""

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
            data += f.read() + "\n"

print(f"length of dataset in characters: {len(data):,}")

data = data.encode('ascii', 'ignore').decode('ascii')

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(x):
    return [stoi[c] for c in x]

def decode(x):
    return ''.join([itos[i] for i in x])

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)