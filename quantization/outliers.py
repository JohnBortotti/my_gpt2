import torch

X = torch.randn(2, 3)
X[:, 2] = 25.0

W = torch.randn(3, 4)

print(f"X: {X}")
print(f"W: {W}")
print()

X_out = X[:, [2]]
W_out = W[[2], :]

print(f"Outliers matrix:\n {X_out} \n {W_out}\n")

X_reg = X[:, (0, 1)]
W_reg = W[(0, 1), :]

y = X @ W
y_dec = X_out @ W_out + X_reg @ W_reg


print(f"y: {y} \ndecomposed y: {y_dec}\n")
