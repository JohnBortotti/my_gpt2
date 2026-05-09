import torch
from absmax_zeropoint import absmax_quantize


def naive_int8_matmul(X, W):
    _, X_deq = absmax_quantize(X)
    _, W_deq = absmax_quantize(W)

    return X_deq @ W_deq

def llm_int8_matmul(X, W):
    threshold = 6.0

    outlier_mask = (X.abs() > threshold).any(dim=0)
    outlier_cols = outlier_mask.nonzero(as_tuple=True)[0]
    non_outlier_cols = (~outlier_mask).nonzero(as_tuple=True)[0]

    X_out = X[:, outlier_cols]
    W_out = W[outlier_cols, :]

    X_reg = X[:, non_outlier_cols]
    W_reg = W[non_outlier_cols, :]

    y = X_out @ W_out + naive_int8_matmul(X_reg, W_reg)

    return y


X = torch.randn(2, 3)
X[:, 2] = 25.0
W = torch.randn(3, 4)


print(f"full precision:\n {X @ W}\n")
print(f"naive int8 matmul:\n {naive_int8_matmul(X, W)}\n")
print(f"llm.int8:\n {llm_int8_matmul(X, W)}\n")

print(f"naive int8 error:\n {(X @ W - naive_int8_matmul(X, W)).abs().mean():.4f}")
print(f"llm.int8 error:\n {(X @ W - llm_int8_matmul(X, W)).abs().mean():.4f}")