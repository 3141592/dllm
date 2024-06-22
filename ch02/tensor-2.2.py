import torch
from torch.nn.utils.rnn import pad_sequence

a = torch.tensor([1, 2, 5, 7])
b = torch.tensor([2, 7, 9])
c = torch.tensor([5, 2])

print("a: ", a)
print("b: ", b)
print("c: ", c)

padded = pad_sequence([a, b, c], batch_first=True)
print("padded: ", padded)

padded_mask = padded.eq(0)
print("padded_mask: ", padded_mask)

print("padded.transpose(0, 1): ", padded.transpose(0, 1))
print("padded.transpose(0, 1).eq(0): ", padded.transpose(0, 1).eq(0))

